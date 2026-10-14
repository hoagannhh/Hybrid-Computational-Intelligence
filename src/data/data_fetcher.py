import os
import pandas as pd
import yfinance as yf
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    source: str = "yfinance",
    save_path: str = "data/raw"
) -> pd.DataFrame:
    """
    Thu thập dữ liệu giá chứng khoán lịch sử (Open, High, Low, Close, Volume).
    
    Args:
        ticker: Mã chứng khoán (ví dụ: 'AAPL', '^GSPC', 'VCB')
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
        source: Nguồn dữ liệu ('yfinance', 'vnstock', 'csv')
        save_path: Thư mục lưu dữ liệu thô
        
    Returns:
        pd.DataFrame: Bảng dữ liệu OHLCV
    """
    logger.info(f"Đang tải dữ liệu cho mã {ticker} từ {start_date} đến {end_date} (Nguồn: {source})...")
    df = pd.DataFrame()

    if source.lower() == "yfinance":
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Làm phẳng multi-index column nếu có
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    elif source.lower() == "vnstock":
        # Placeholder nếu người dùng sử dụng thư viện vnstock cho thị trường chứng khoán Việt Nam
        try:
            from vnstock import stock_historical_data
            df = stock_historical_data(ticker=ticker, start_date=start_date, end_date=end_date)
        except ImportError:
            logger.error("Chưa cài đặt thư viện 'vnstock'. Hãy cài qua: pip install vnstock")
            raise
    elif source.lower() == "csv":
        csv_file = os.path.join(save_path, f"{ticker}.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"Không tìm thấy file: {csv_file}")
    else:
        raise ValueError(f"Nguồn dữ liệu không hợp lệ: {source}")

    if df.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho mã: {ticker}")

    # Đảm bảo index là DateTime và sắp xếp tăng dần
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Lưu dữ liệu thô vào thư mục data/raw
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"{ticker.replace('^', '')}_{start_date}_{end_date}.csv")
    df.to_csv(out_file)
    logger.info(f"Dữ liệu đã được lưu thành công tại: {out_file} (Tổng số phiên: {len(df)})")

    return df
