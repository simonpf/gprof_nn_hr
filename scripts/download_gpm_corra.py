"""
Script to download a month of GPM combined radar-radiometer (CORRA) retrievals
using pansat.

Available files of the 'l2b_corra2022_gpm_dprgmi_v07c' product are looked up
for each day of the month using 'find_files' and subsequently downloaded using
one or multiple threads. Files that fail to download are written to a text
file in the destination folder so that they can be retried later.

Usage:
    python download_gpm_corra.py 2025 1 --n_threads 4
"""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time

from pansat.download.accounts import get_identity
import pansat.environment as penv
from pansat.products.satellite.gpm import l2b_corra2022_gpm_dprgmi_v07c
from pansat.time import TimeRange
from tqdm import tqdm

LOGGER = logging.getLogger(__file__)

PRODUCT = l2b_corra2022_gpm_dprgmi_v07c

######################################################################
# Arguments
######################################################################

parser = argparse.ArgumentParser(
    description="Download a month of GPM CORRA retrievals using pansat."
)
parser.add_argument("year", metavar="year", type=int, help="The year to download.")
parser.add_argument(
    "month", metavar="month", type=int, help="The month to download (1 - 12)."
)
parser.add_argument(
    "--destination",
    metavar="path",
    type=str,
    default=None,
    help=(
        "The root folder in which to store the data. Defaults to the active "
        "pansat data directory. Files are stored in "
        f"'<destination>/{PRODUCT.default_destination}'."
    ),
)
parser.add_argument(
    "--n_threads",
    metavar="n",
    type=int,
    default=1,
    help="Number of threads to use for download.",
)
parser.add_argument(
    "--n_retries",
    metavar="n",
    type=int,
    default=3,
    help="Number of times to retry a failed file search or download.",
)

args = parser.parse_args()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

if args.destination is None:
    root = penv.get_active_data_dir()
else:
    root = Path(args.destination)
root.mkdir(exist_ok=True, parents=True)


######################################################################
# Find files
######################################################################


def find_files_for_day(product, start, n_retries):
    """
    Find available files of a product for a single day, retrying on failure.
    """
    time_range = TimeRange(start, start + timedelta(days=1))
    for attempt in range(n_retries + 1):
        try:
            return product.find_files(time_range)
        except Exception:
            if attempt == n_retries:
                raise
            LOGGER.warning(
                "Finding files of product %s for %s failed. Retrying %s more "
                "time(s).",
                product.name,
                start.date(),
                n_retries - attempt,
            )
            time.sleep(2**attempt)


def find_files(product, year, month, n_retries):
    """
    Find available files of a product for every day of a given month.

    Return:
        A tuple ``(recs, failed_days)`` containing the file records of all
        unique files and the days for which the file search failed.
    """
    recs = {}
    failed_days = []
    n_days = calendar.monthrange(year, month)[1]
    for day in range(1, n_days + 1):
        start = datetime(year, month, day)
        try:
            day_recs = find_files_for_day(product, start, n_retries)
        except Exception:
            LOGGER.exception(
                "Finding files of product %s for %s failed.",
                product.name,
                start.date(),
            )
            failed_days.append(start.date())
            continue
        LOGGER.info(
            "Found %s files of product %s for %s.",
            len(day_recs),
            product.name,
            start.date(),
        )
        # Granules that cross midnight are found for both days.
        for rec in day_recs:
            recs.setdefault(rec.filename, rec)
    return list(recs.values()), failed_days


######################################################################
# Download files
######################################################################


def download_file(rec, destination, n_retries):
    """
    Download a file and remove partially downloaded files on failure.
    """
    path = destination / rec.filename
    if path.exists():
        LOGGER.debug("Skipping existing file %s.", path)
        return path

    for attempt in range(n_retries + 1):
        try:
            rec.download(destination=destination)
            LOGGER.debug("Downloaded %s.", path)
            return path
        except Exception:
            path.unlink(missing_ok=True)
            if attempt == n_retries:
                raise
            LOGGER.warning(
                "Downloading %s failed. Retrying %s more time(s).",
                rec.filename,
                n_retries - attempt,
            )


# Retrieving the GES DISC identity requires the pansat password. Doing this
# once before starting the download threads ensures that the user is prompted
# for the password only once and that a wrong password or a missing identity
# is detected before the file search.
get_identity("GES DISC")

destination = root / PRODUCT.default_destination
destination.mkdir(exist_ok=True, parents=True)

recs, failed_days = find_files(PRODUCT, args.year, args.month, args.n_retries)
LOGGER.info("Downloading %s files of product %s.", len(recs), PRODUCT.name)

failed_files = []
with ThreadPoolExecutor(max_workers=args.n_threads) as pool:
    tasks = {
        pool.submit(download_file, rec, destination, args.n_retries): rec
        for rec in recs
    }
    for task in tqdm(as_completed(tasks), total=len(tasks), unit="file"):
        rec = tasks[task]
        try:
            task.result()
        except Exception:
            LOGGER.exception("Downloading %s failed.", rec.filename)
            failed_files.append(rec)

if failed_files:
    failed_path = root / f"failed_gpm_corra_{args.year}_{args.month:02}.txt"
    with open(failed_path, "w") as output:
        for rec in failed_files:
            output.write(f"{rec.remote_path}\n")
    LOGGER.warning(
        "Downloading %s files failed. Their URLs were written to %s.",
        len(failed_files),
        failed_path,
    )

for day in failed_days:
    LOGGER.warning("Finding files for %s failed.", day)

if not failed_files and not failed_days:
    LOGGER.info("All files downloaded successfully.")
