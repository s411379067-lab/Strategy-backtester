from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from ..models import OrderIntent, ActionType, Side, ExitType, Position, SizingEquityBase
from ..strategy_base import Strategy, StrategyContext
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class StrongBreakoutParams:
    """
    強勢突破策略參數
    
    ---
    - BBR_thresholds: Tuple[float, float] = lambda: (0.4, 0.7)  # K線實體占比
    - close_loc_thresholds: Tuple[float, float] = lambda: (0.6, 0.8)  # 收盤價在當根K棒的位置(陽線上0.6，陰線下0.6)
    - push_thresholds: Tuple[float, float, float] = lambda: (0.5, 1.0, 1.5)  # K線漲跌幅度為ATR的多少倍
    - overlap_thresholds: Tuple[float, float] = lambda: (0.3, 0.5)  # 突破段內重疊的程度（重疊區間占段長的比例）
    - break_out_n_bars: int = 20  # 計算突破用的n（例如前n根K線）
    - run_score_threshold: float = 5.0  # Run Score的分數門檻，高於分數門檻才考慮進場
    - session: Optional[str] = None  # 可選的交易時段，例如 "Asian", "EU", "US"，如果為None表示全天交易
    - breakout_based_on: str = "close"  # 突破基於哪個價格，"close"或"extreme"（最高點/最低點）
    

    - max_equity_loss_pct: float = 1.0  # 單筆交易最大損失占比（%）
    - min_qty: float = 0.001  # 最小交易量
    - min_sl_pct: float = 0.5  # 最小停損距離占比（%），如果計算出來的停損距離太小，則不進行交易

    - rr: float = 2.0           # 風險回報比，TP距離 = SL距離 * rr
    - time_exit_bars: int = 50  # 時間出場條件
    - allow_side: Optional[Side] = None  # 允許的交易方向，None表示雙向進出場，Side.LONG表示只做多，Side.SHORT表示只做空
    - sizing_equity_base: SizingEquityBase = SizingEquityBase.INITIAL  # 倉位大小計算基礎，INITIAL表示以初始資金為基礎，CURRENT表示以當前資金為基礎

    """
    BBR_thresholds: Tuple[float, float] = lambda: (0.4, 0.7)
    close_loc_thresholds: Tuple[float, float] = lambda: (0.6, 0.8)
    push_thresholds: Tuple[float, float, float] = lambda: (0.5, 1.0, 1.5)
    overlap_thresholds: Tuple[float, float] = lambda: (0.3, 0.5)

    break_out_n_bars: int = 20  # 計算突破用的n（例如前n根K線）

    run_score_threshold: float = 5.0
    session: Optional[str] = None  # 可選的交易時段，例如 "Asian", "European", "US"，如果為None表示全天交易
    breakout_based_on: str = "close"  # 突破基於哪個價格，"close"或"extreme"（最高點/最低點）



    max_equity_loss_pct: float = 1.0
    min_qty: float = 0.001
    min_sl_pct: float = 0.5

    rr: float = 2.0           # TP = SL距離 * rr
    time_exit_bars: int = 50
    allow_side: Optional[Side] = None  # None表示雙向進出場，Side.LONG表示只做多，Side.SHORT表示只做空
    sizing_equity_base: SizingEquityBase = SizingEquityBase.INITIAL



class StrongBreakoutStrategy(Strategy):
    def __init__(self, params: StrongBreakoutParams) -> None:
        self.p = params
        self.state = None  # 可用來記錄策略狀態
        self.pb_start_i = None  # 紀錄pullback開始的index
        self.tp_changed = False  # 是否已經移動過停利
        self.last_hh = None # 紀錄最近一次的最高點


    def required_indicators(self) -> Dict[str, Any]:
        return {
            "bar_score_df": ("K_bar_score", self.p.BBR_thresholds, self.p.close_loc_thresholds, self.p.push_thresholds, self.p.overlap_thresholds),
            "run_score_df": ("K_run_score", self.p.BBR_thresholds, self.p.close_loc_thresholds, self.p.push_thresholds, self.p.overlap_thresholds),
            "hh": ("rolling_high", self.p.break_out_n_bars, "high"),
            "ll": ("rolling_low", self.p.break_out_n_bars, "low"),
        }

    def generate_intents(self, ctx: StrategyContext) -> List[OrderIntent]:
        i = ctx.i
        df = ctx.df
        pos = ctx.position
        init_equity = ctx.init_equity
        now_equity = ctx.now_equity
        if self.p.sizing_equity_base == SizingEquityBase.INITIAL:
            base_equity = init_equity
        elif self.p.sizing_equity_base == SizingEquityBase.CURRENT:
            base_equity = now_equity
        intents: List[OrderIntent] = []

        open_series = df["open"]
        high_series = df["high"]
        low_series = df["low"]
        close_series = df["close"]
        close_p = float(close_series.iat[i])
        high_p = float(high_series.iat[i])
        low_p = float(low_series.iat[i])

        # 若有倉，只更新出場線（也可以不更新）
        if pos.side is not None and pos.qty > 0:

            info = pos.position_info or {}
            bar_score_i = float(ctx.indicators["bar_score_df"]["bar_score"].iat[i])

            k = i - pos.entry_bar_i  # 進場後第幾根

            patch = {}

            # 進場後第 1~3 根累加
            if 1 <= k <= 3:
                sum3 = float(info.get("sum_3bar_score_after_entry", 0.0)) + bar_score_i
                patch["sum_3bar_score_after_entry"] = sum3

            # 只在進場下一根記錄 follow-through
            if k == 1:
                patch["follow_through_score"] = bar_score_i

            # 如果 patch 不是空的才送 UPDATE（避免每根都送）
            if patch:
                intents.append(OrderIntent(
                    action=ActionType.UPDATE,
                    side=pos.side,
                    qty=0.0,
                    position_info=patch,
                    priority=20,
                ))

            return intents


        # 無倉：
        # 做突破進場（LONG）
        ## 亞洲盤
        time_hour = df.index[i].hour
        if self.p.session == "Asian":
            time_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif self.p.session == "EU":
            time_session = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        elif self.p.session == "US":
            time_session = [13, 14, 15, 16, 17, 18, 19, 20, 21]
        else:
            time_session = list(range(24)) # 全天交易
        
        cond_time_filter = time_hour in time_session

        if i < 1:
            return intents
        
        bar_score_df = ctx.indicators["bar_score_df"]
        run_score_df = ctx.indicators["run_score_df"]
        run_score_series = run_score_df["run_score"]
        seg_id_series = run_score_df["seg_id"]
        bar_score_series = bar_score_df["bar_score"]

        seg_id = seg_id_series.iat[i]
        # i為int， 找出到i為止的數據
        seg_index_before_i = seg_id_series.iloc[:i+1]
        seg_index_series = seg_index_before_i[seg_index_before_i == seg_id].index
        

        # print(f"i={i}, seg_id={seg_id}, seg_index={seg_index_series.tolist()}")


        # 分數條件
        run_score_cond = run_score_series.iat[i] >= self.p.run_score_threshold
        run_score_short_cond = run_score_series.iat[i] <= -self.p.run_score_threshold

        # 突破條件（也可以放在分數條件前面）
        hh = ctx.indicators["hh"]
        ll = ctx.indicators["ll"]
        hh_i = hh.iat[i-1]
        ll_i = ll.iat[i-1]
        if self.p.breakout_based_on == "close":
            breakout_cond = close_p > hh_i
            breakout_short_cond = close_p < ll_i
        elif self.p.breakout_based_on == "extreme":
            breakout_cond = high_p > hh_i
            breakout_short_cond = low_p < ll_i
        else:
            raise ValueError(f"Invalid breakout_based_on value: {self.p.breakout_based_on}")

        # 方向條件
        side_cond = self.p.allow_side is None or self.p.allow_side == Side.LONG
        short_side_cond = self.p.allow_side is None or self.p.allow_side == Side.SHORT

        if run_score_cond and side_cond and breakout_cond and cond_time_filter:
            # 收盤進場
            entry_price = float(close_series.iat[i])  
            # seg第一根低點當作止損（也可以改成其他邏輯）
            # seg_low = low_series.loc[seg_index_series].min()
            # seg內 bar score最高的那根的低點當作止損（分數相同時取比較早出現的那根)
            seg_bar_score = bar_score_series.loc[seg_index_series]
            # seg_bar_score = seg_bar_score.sort_values(ascending=False, kind="stable")  # 分數高的在前面

            # 取最高分、如果SL太大就取第二高分
            # for idx, score in seg_bar_score.items():
            #     seg_low = low_series.loc[idx]
            #     sl_price = seg_low
            #     sl_range = entry_price - sl_price
            #     sl_range_pct = sl_range / entry_price * 100 if entry_price > 0 else 0.0
            #     tp_price = entry_price + sl_range * self.p.rr
            #     if sl_range_pct < self.p.min_sl_pct:
            #         break  # 找到第一個符合停損距離要求的bar就停止

            # 直接取最高分的那根，停損距離如果太大就放棄這筆交易
            candidate_index = seg_bar_score.idxmax()
            seg_low = low_series.loc[candidate_index]
            sl_price = seg_low
            sl_range = entry_price - sl_price
            tp_price = entry_price + sl_range * self.p.rr
            # 計算倉位大小（簡化：以初始資金為基礎）
            max_notional_loss = base_equity * self.p.max_equity_loss_pct / 100
            qty = max_notional_loss / sl_range if sl_range > 0 else self.p.min_qty

            sl_range_pct = sl_range / entry_price * 100 if entry_price > 0 else 0.0
            if sl_range_pct > self.p.min_sl_pct:
                return intents

            intents.append(
                OrderIntent(
                    action=ActionType.ENTRY,
                    side=Side.LONG,
                    qty=max(qty, self.p.min_qty),
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    be_price=None,
                    position_info={"follow_through_score": None, "entry_bar_score": bar_score_series.iat[i], "sum_3bar_score_after_entry": 0},  # 可以記錄進場的index，後續用來判斷持倉時間等
                    priority=10,
                )
            )
        elif run_score_short_cond and short_side_cond and breakout_short_cond and cond_time_filter:
            entry_price = float(close_series.iat[i])  
            seg_bar_score = bar_score_series.loc[seg_index_series]
            candidate_index = seg_bar_score.idxmin()
            seg_high = high_series.loc[candidate_index]
            sl_price = seg_high
            sl_range = sl_price - entry_price
            tp_price = entry_price - sl_range * self.p.rr

            max_notional_loss = base_equity * self.p.max_equity_loss_pct / 100
            qty = max_notional_loss / sl_range if sl_range > 0 else self.p.min_qty

            sl_range_pct = sl_range / entry_price * 100 if entry_price > 0 else 0.0
            if sl_range_pct > self.p.min_sl_pct:
                return intents

            intents.append(
                OrderIntent(
                    action=ActionType.ENTRY,
                    side=Side.SHORT,
                    qty=max(qty, self.p.min_qty),
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    be_price=None,
                    position_info={"follow_through_score": None, "entry_bar_score": bar_score_series.iat[i], "sum_3bar_score_after_entry": 0},  # 可以記錄進場的index，後續用來判斷持倉時間等
                    priority=10,
                )
            )


        return intents

