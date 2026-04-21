# BYTE

A repository for scraping real estate data from Yandex Realty using Selenium and BeautifulSoup.

## Description

This project contains a Python script in a Jupyter Notebook (ya_parser_raw.ipynb) that parses apartment listings from Yandex Realty. It uses Selenium for browser automation to handle dynamic content and captchas, extracts details like price, area, rooms, floor, metro, address, and more, and saves the data to a CSV file.

The parsed data is stored in yandex_realty_manual.csv.

## Features

- Parses multiple pages with configurable parameters (e.g., area range, floor range).
- Handles captchas and waits for page loads.
- Extracts structured data from offer snippets.
- Supports incremental parsing by loading existing CSV data.
- Saves data periodically to avoid data loss.

## Requirements

- Python 3.x
- Google Chrome browser
- ChromeDriver (managed by Selenium)

## Dependencies

Install the required packages using pip:

```
pip install pandas selenium beautifulsoup4 fake-useragent lxml
```

## Usage

1. Open the Jupyter Notebook ya_parser_raw.ipynb.
2. Run the cells to execute the parser.
3. The script will start Chrome, navigate to Yandex Realty, and parse listings based on the defined grid parameters.
4. Data is saved to yandex_realty_manual.csv.

### Key Parameters

- `BASE_URL`: The starting URL for Moscow and Moscow Oblast apartments.
- `GRID_PARAMS`: List of parameter dictionaries for filtering (e.g., areaMin, areaMax, floorMin, floorMax).
- `pages_per_param`: Number of pages to parse per parameter set (default: 25).
- `save_interval`: Save to CSV every N pages (default: 5).

### Example

```python
parser = YandexRealtyParser(headless=False, existing_csv_path=OUTPUT_CSV)
parser.start()
try:
    parser.parse_grid(BASE_URL, GRID_PARAMS, pages_per_param=25, save_interval=5)
finally:
    parser.close()
```

## Data Structure

The CSV contains columns such as:

- `offer_id`: Unique offer identifier.
- `price`: Price string (e.g., "10 000 000 ₽").
- `price_numeric`: Numeric price.
- `area`: Area in m².
- `rooms`: Number of rooms or "студия".
- `floor`: Floor (e.g., "5/10").
- `metro`: Nearest metro station.
- `address`: Full address.
- `author`: Seller type.
- `main_image`: URL of main image.
- `photo_count`: Number of photos.
- `badges`: Features like "новостройка".
- `publish_date`: Date published.
- `url`: Offer URL.
- `description`: Description text.
- `image_urls`: Semicolon-separated image URLs.

## Notes

- The parser uses a fake user agent and anti-detection measures to mimic a real browser.
- Be respectful of website terms of service and avoid overloading the server.
- For headless mode, set `headless=True` in `YandexRealtyParser`.
- If captchas appear, the script waits for manual solving (timeout: 300 seconds).

## License

This project is for educational purposes. Check Yandex Realty's terms before use.