import gzip
import json
import subprocess
import io
from typing import Optional, Literal

import numpy as np
from numpy.typing import NDArray

from hftbacktest.data.validation import correct_event_order, correct_local_timestamp, validate_event_order
from hftbacktest.types import (
    DEPTH_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    TRADE_EVENT,
    BUY_EVENT,
    SELL_EVENT,
    event_dtype
)


def convert(
        input_filename: str,
        output_filename: Optional[str] = None,
        opt: Literal['', 'm', 't', 'mt'] = '',
        base_latency: float = 0,
        combined_stream: bool = True,
        buffer_size: int = 100_000_000_000
) -> NDArray:
    r"""
    Converts raw Binance WebSocket feed stream file into a format compatible with HftBacktest.
    If you encounter an ``IndexError`` due to an out-of-bounds, try increasing the ``buffer_size``.

    **File Format:**

    .. code-block::

       1751240969020830000 {"stream":"xrpusdt@bookTicker","data":{"e":"bookTicker","u":7910652949033,"s":"XRPUSDT","b":"2.2058","B":"8122.6","a":"2.2059","A":"31314.9","T":1751240968948,"E":1751240968948}}
       1751240969140033000 {"stream":"xrpusdt@depth@0ms","data":{"e":"depthUpdate","E":1751240968961,"T":1751240968953,"s":"XRPUSDT","U":7910652947517,"u":7910652949212,"pu":7910652947340,"b":[["2.2005","102082.0"]],"a":[["2.2059","31314.9"]]}}

    Args:
        input_filename: Input filename with path.
        output_filename: If provided, the converted data will be saved to the specified filename in ``npz`` format.
        opt: Additional processing options (currently unused for Binance format).
        base_latency: The value to be added to the feed latency.
                      See :func:`.correct_local_timestamp`.
        combined_stream: Raw stream type (currently unused for Binance format).
        buffer_size: Sets a preallocated row size for the buffer.

    Returns:
        Converted data compatible with HftBacktest.
    """
    timestamp_slice = 19

    tmp = np.empty(buffer_size, event_dtype)
    row_num = 0
    
    # Use gunzip command to handle potentially corrupted gzip files
    print("Using gunzip command to decompress file...")
    process = subprocess.Popen(['gunzip', '-c', input_filename], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
    
    if process.stdout is None:
        raise RuntimeError("Failed to start gunzip process")
    
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse timestamp and JSON message - 确保使用正确的数据类型
                local_timestamp = np.int64(line[:timestamp_slice])
                message = json.loads(line[timestamp_slice + 1:])
                
                stream = message.get('stream', '')
                data = message.get('data')
                
                if data is None:
                    continue
                
                event_type = data.get('e', '')
                # Binance的E和T字段是毫秒时间戳，转换为纳秒需要乘以1_000_000
                exch_timestamp = np.int64(data.get('E', 0)) * np.int64(1_000_000)  # Convert milliseconds to nanoseconds
                
                # Handle depth updates
                if event_type == 'depthUpdate':
                    bids = data.get('b', [])
                    asks = data.get('a', [])
                    
                    # Process bids
                    for px_str, qty_str in bids:
                        px = float(px_str)
                        qty = float(qty_str)
                        
                        tmp[row_num] = (
                            DEPTH_EVENT | BUY_EVENT,
                            exch_timestamp,
                            local_timestamp,
                            px,
                            qty,
                            0,
                            0,
                            0
                        )
                        row_num += 1
                        
                        if row_num >= buffer_size - 1000:
                            break
                    
                    # Process asks
                    for px_str, qty_str in asks:
                        px = float(px_str)
                        qty = float(qty_str)
                        
                        tmp[row_num] = (
                            DEPTH_EVENT | SELL_EVENT,
                            exch_timestamp,
                            local_timestamp,
                            px,
                            qty,
                            0,
                            0,
                            0
                        )
                        row_num += 1
                        
                        if row_num >= buffer_size - 1000:
                            break
                
                # Handle trade data
                elif event_type == 'trade':
                    price = float(data.get('p', '0'))
                    quantity = float(data.get('q', '0'))
                    is_buyer_maker = data.get('m', False)
                    # Binance的T字段是毫秒时间戳，转换为纳秒需要乘以1_000_000
                    trade_time = np.int64(data.get('T', 0)) * np.int64(1_000_000)  # Convert milliseconds to nanoseconds
                    
                    # Determine side: if buyer is maker, then it's a sell trade (taker sold)
                    # if seller is maker, then it's a buy trade (taker bought)
                    side = SELL_EVENT if is_buyer_maker else BUY_EVENT
                    
                    tmp[row_num] = (
                        TRADE_EVENT | side,
                        trade_time,
                        local_timestamp,
                        price,
                        quantity,
                        0,
                        0,
                        0
                    )
                    row_num += 1
                
                # Handle book ticker (best bid/ask updates)
                elif event_type == 'bookTicker':
                    # We can optionally process bookTicker events as depth updates
                    # But for now, we'll skip them to avoid duplicate data
                    pass
                    
            except (json.JSONDecodeError, ValueError, KeyError, OverflowError) as e:
                # Skip invalid lines, including overflow errors
                print(f"Skipping invalid line: {e}")
                continue
                
            # Check if buffer is getting full
            if row_num >= buffer_size - 1000:  # Leave some buffer space
                print(f"Warning: Buffer nearly full at {row_num}/{buffer_size}. Consider increasing buffer_size.")
                break
                
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up subprocess
        process.wait()
        if process.returncode != 0 and process.stderr:
            stderr_output = process.stderr.read()
            if stderr_output.strip():  # Only print if there's actual error content
                print(f"gunzip command failed: {stderr_output}")

    tmp = tmp[:row_num]

    print('Correcting the latency')
    tmp = correct_local_timestamp(tmp, base_latency)

    print('Correcting the event order')
    data = correct_event_order(
        tmp,
        np.argsort(tmp['exch_ts'], kind='mergesort'),
        np.argsort(tmp['local_ts'], kind='mergesort')
    )

    validate_event_order(data)

    if output_filename is not None:
        print('Saving to %s' % output_filename)
        np.savez_compressed(output_filename, data=data)

    return data
