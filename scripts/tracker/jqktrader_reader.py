"""jqktrader integration: read positions from 同花顺 client.

This is a placeholder module. Actual functionality requires:
1. 同花顺客户端 installed and running
2. jqktrader library: pip install jqktrader
3. See: https://github.com/zsrl/jqktrader

Usage (when available):
    reader = JQKTraderReader()
    positions = reader.read_positions()
    # Then pass to PortfolioTracker.save_snapshot()
"""
from typing import Optional


class JQKTraderReader:
    """Read portfolio positions from 同花顺 via jqktrader.

    This is a placeholder implementation. Install jqktrader and
    configure the connection to enable automatic position reading.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._client = None

    def _get_client(self):
        """Lazy-init the jqktrader client."""
        if self._client is not None:
            return self._client

        try:
            import jqktrader
            # jqktrader API — adjust based on actual library version
            # self._client = jqktrader.Client(...)
            raise NotImplementedError(
                "jqktrader integration not yet configured. "
                "Please install jqktrader and update this module."
            )
        except ImportError:
            raise ImportError(
                "jqktrader 未安装。请运行: pip install jqktrader\n"
                "参考: https://github.com/zsrl/jqktrader"
            )

    def read_positions(self) -> list[dict]:
        """Read current holdings from 同花顺.

        Returns:
            List of dicts with standard format:
            [{stock_code, stock_name, shares, cost_price, current_price,
              market_value, pnl_amount, pnl_pct}, ...]
        """
        try:
            client = self._get_client()
            # Placeholder — actual API calls depend on jqktrader version
            # raw = client.get_positions()
            # return [self._normalize(r) for r in raw]
            return []
        except (ImportError, NotImplementedError) as e:
            print(f"⚠️ {e}")
            return []

    def read_capital(self) -> Optional[dict]:
        """Read capital info from 同花顺.

        Returns:
            Dict with: total_assets, available_cash, market_value, total_pnl
        """
        try:
            client = self._get_client()
            # Placeholder
            return None
        except (ImportError, NotImplementedError):
            return None

    def _normalize(self, raw: dict) -> dict:
        """Normalize jqktrader output to our standard format."""
        return {
            "stock_code": raw.get("证券代码", ""),
            "stock_name": raw.get("证券名称", ""),
            "shares": int(raw.get("股票余额", 0)),
            "cost_price": float(raw.get("成本价", 0)),
            "current_price": float(raw.get("最新价", 0)),
            "market_value": float(raw.get("股票市值", 0)),
            "pnl_amount": float(raw.get("浮动盈亏", 0)),
            "pnl_pct": float(raw.get("盈亏比例", 0)),
        }

    def is_available(self) -> bool:
        """Check if jqktrader is installed and ready."""
        try:
            import jqktrader
            return True
        except ImportError:
            return False
