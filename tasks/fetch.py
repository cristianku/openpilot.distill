"""Copy segments of the comma into the shared folder of the comma files and link them into data/.

    python -m tasks.fetch [--host=<ip|name>] [--since=<YYYY-MM-DD>] [--localizer]
    python -m tasks.fetch [--host=<ip|name>] <route> --segments=<list> [--localizer]
    python -m tasks.fetch [--host=<ip|name>] <route> --skip=<list> [--localizer]
    python -m tasks.fetch [--host=<ip|name>] <route> [--strip-ends] [--localizer]
  python -m tasks.fetch                                           (automatic, see below)
  python -m tasks.fetch --since=2026-10-25
  python -m tasks.fetch 6616faac453a3064/0000011c--8152ba83cf     (= --strip-ends)
  python -m tasks.fetch 6616faac453a3064/0000011c--8152ba83cf --skip=0,1,2,last
  python -m tasks.fetch 6616faac453a3064/0000011c--8152ba83cf --skip 0,1,14
  python -m tasks.fetch 6616faac453a3064/0000011c--8152ba83cf --segments=1,2,5-9
  python -m tasks.fetch 6616faac453a3064/0000011c--8152ba83cf/1   (connect format)
  python -m tasks.fetch --host 192.168.88.29 6616faac453a3064/0000011c--8152ba83cf/1
  python -m tasks.fetch 0000011c--8152ba83cf 0 1 2
  python -m tasks.fetch 0000011c--8152ba83cf 2-9

Where the files go: rlog.zst, fcamera.hevc and ecamera.hevc are downloaded once into the shared
folder (COMMA_VIDEOS_DIR, default comma-videos next to this repo, e.g. /root/comma-videos), one
folder per segment as on the comma; sunnypilot.speedlimits uses the same files. data/<folder>/ only
has absolute links to them, plus what the localizer writes (never shared).
  <COMMA_VIDEOS_DIR>/0000011c--8152ba83cf--3/fcamera.hevc        (the file)
  data/0000011c8152ba83cf03/fcamera.hevc -> the file above       (the link)
A file still downloaded straight into data/ (old layout) is moved into the shared folder and
replaced by the link; if the shared folder already has a different copy it stops, nothing deleted.
Both segment folders must be real folders: when data/<folder> or the shared one is a link (the two
would be the same files) it stops too, nothing deleted.

Folder name in data/: route without dashes + segment number on two digits
  0000011c--8152ba83cf--3  ->  data/0000011c8152ba83cf03
(supervised/dataset.py does int(segment, 16): dashes would crash it.)

Without a route (automatic): every route on the comma with a segment written on or after --since
(default 2026-10-20) is checked for dashcam mode. Before fetching, whole trips lasting less than 300
seconds are excluded (measured from the very first to the very last qlog, INCLUDING the first/last
two segments). Then the first two + last two segments are discarded, and any remaining segment whose
rlog contains carState.gearShifter=reverse. Missing/unreadable logs are skipped, never assumed safe.
Explicit route/segment selection stays manual and does not apply these automatic filters.
Segments the localizer moved to data_scartati/ are not downloaded again.
The logs are read by helpers/comma_fetch_log.py (the same reader as sunnypilot.speedlimits);
COMMA_FETCH_PYTHON can override the Python interpreter that runs it.
Already downloaded files are skipped by rsync, so it can be run after every drive.

--skip: every segment of the route found on the comma, except the listed ones. The list takes
        numbers, ranges (2-5), "last" (the highest segment, the arrival), "last-N" (N segments
        before the last one) and first, second, second_last (the lowest, the second lowest and
        the second highest segment found).
--strip-ends: skip first two and last two AVAILABLE segment folders (even with gaps). No list
              needed. It is the default when neither segments nor --skip are given.
--localizer: also run the localizer on each segment (off by default: tasks.localize does it later
             and redoes it when a better ionosphere map comes out). --no-localizer turns it off.
--since: automatic mode only, first day to look at (YYYY-MM-DD).
--host: IP or ssh name of the comma; "comma@" is added when no user is given.
Env: COMMA_HOST (default "comma", the Host in ~/.ssh/config), overridden by --host
     COMMA_VIDEOS_DIR (default comma-videos next to this repo), shared folder of the comma files
     Both can also be set in the repo .env.
"""
from __future__ import annotations

import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from tasks.comma import Comma, comma_host
from tasks.common import REPO, TaskError, load_env, run_main, say

REMOTE_DIR = '/data/media/0/realdata'
DATA_DIR = REPO / 'data'
DISCARD_DIR = REPO / 'data_scartati'
FILES = ('rlog.zst', 'fcamera.hevc', 'ecamera.hevc')
DEFAULT_SINCE = '2026-10-20'
MIN_ROUTE_SECONDS = 300
STRIP_ENDS = ('first', 'second', 'second_last', 'last')
SKIP_WORDS = ('first', 'second', 'second_last', 'last')

DATE_RE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
# A route as the comma names it, also as the dataset needs it once the dashes are gone (hex).
ROUTE_RE = re.compile(r'^[0-9a-f]+--[0-9a-f]+$')
# connect format with segment: <dongle_id>/<route>/<segment>
CONNECT_SEGMENT_RE = re.compile(r'^([0-9a-f]+/)?([0-9a-f]+--[0-9a-f]+)/([0-9]+)$')
# route alone, with or without the dongle id: <dongle_id>/<route>
CONNECT_ROUTE_RE = re.compile(r'^([0-9a-f]+/)?([0-9a-f]+--[0-9a-f]+)$')
NUMBER_RE = re.compile(r'^[0-9]+$')
RANGE_RE = re.compile(r'^[0-9]+-[0-9]+$')
LAST_MINUS_RE = re.compile(r'^last-[0-9]+$')
SECONDS_RE = re.compile(r'^[0-9]+([.][0-9]+)?$')

# The segment folders of the comma with their time: "2026-10-20 14:03:22.123456789 +0000 <folder>".
LISTING_COMMAND = 'cd ' + REMOTE_DIR + ' && stat -c \'%y %n\' *--*--* 2>/dev/null || true'
# The segment folders of one route. $1 = the route.
ROUTE_SEGMENTS_SCRIPT = 'ls -d ' + REMOTE_DIR + '/"$1"--* 2>/dev/null || true\n'
# $1 = a path on the comma: exit code 0 when it is a file.
FILE_EXISTS_SCRIPT = 'test -f "$1"\n'


def warn(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


@dataclass
class Options:
    host: str | None = None
    since: str = DEFAULT_SINCE
    route: str = ''
    segments: list[int] = field(default_factory=list)
    # Numbers, and the words first, second, second_last, last, last-N (turned into numbers only
    # after asking the comma which segments it has: resolve_skip).
    skip: list = field(default_factory=list)
    skip_mode: bool = False
    localizer: bool = False
    auto: bool = False
    help: bool = False


def parse_segment(item: str) -> list[int]:
    """One segment item: a number (3) or a range (2-9)."""
    if RANGE_RE.fullmatch(item):
        first, last = item.split('-')
        return list(range(int(first), int(last) + 1))
    if NUMBER_RE.fullmatch(item):
        return [int(item)]
    raise TaskError('invalid segment: ' + item)


def _list_items(text: str) -> list[str]:
    # As bash "IFS=, read -a": a final empty item ("1,2,") is not an item.
    items = text.split(',')
    if items and items[-1] == '':
        items.pop()
    return items


def parse_segment_list(text: str) -> list[int]:
    """A comma separated list (1,2,5-9)."""
    if not text:
        raise TaskError('--segments needs a list (e.g. --segments=1,2,3)')
    segments = []
    for item in _list_items(text):
        segments += parse_segment(item)
    return segments


def parse_skip_list(text: str) -> list:
    """A comma separated skip list (0,1,14 or 0,1,2,last or 2-5); the words stay words."""
    if not text:
        raise TaskError('--skip needs a list (e.g. --skip=0,1,last)')
    skip = []
    for item in _list_items(text):
        if item in SKIP_WORDS or LAST_MINUS_RE.fullmatch(item):
            skip.append(item)
        elif RANGE_RE.fullmatch(item) or NUMBER_RE.fullmatch(item):
            skip += parse_segment(item)
        else:
            raise TaskError('invalid segment to skip: ' + item)
    return skip


def _check_since(value: str) -> str:
    if not DATE_RE.fullmatch(value):
        raise TaskError('--since needs a YYYY-MM-DD date (e.g. --since=2026-10-20)')
    return value


def parse_args(arguments: list[str]) -> Options:
    """The options, read in order as the arguments come (a later --host wins, lists add up)."""
    options = Options()
    index = 0

    def value_of_next() -> str:
        nonlocal index
        index += 1
        return arguments[index] if index < len(arguments) else ''

    while index < len(arguments):
        argument = arguments[index]
        if argument in ('-h', '--help'):
            options.help = True
            return options
        if argument == '--since':
            options.since = _check_since(value_of_next())
        elif argument.startswith('--since='):
            options.since = _check_since(argument[len('--since='):])
        elif argument == '--localizer':
            options.localizer = True
        elif argument == '--no-localizer':
            options.localizer = False
        elif argument == '--strip-ends':
            options.skip += list(STRIP_ENDS)
            options.skip_mode = True
        elif argument == '--skip' or argument.startswith('--skip='):
            text = value_of_next() if argument == '--skip' else argument[len('--skip='):]
            options.skip += parse_skip_list(text)
            options.skip_mode = True
        elif argument == '--host' or argument.startswith('--host='):
            host = value_of_next() if argument == '--host' else argument[len('--host='):]
            if not host:
                raise TaskError('--host needs an IP or a name')
            options.host = host
        elif argument == '--segments':
            options.segments += parse_segment_list(value_of_next())
        elif argument.startswith('--segments='):
            options.segments += parse_segment_list(argument[len('--segments='):])
        elif not options.route and CONNECT_SEGMENT_RE.fullmatch(argument):
            match = CONNECT_SEGMENT_RE.fullmatch(argument)
            options.route = match.group(2)
            options.segments.append(int(match.group(3)))
        elif not options.route and CONNECT_ROUTE_RE.fullmatch(argument):
            options.route = CONNECT_ROUTE_RE.fullmatch(argument).group(2)
        elif not options.route:
            raise TaskError('invalid route: ' + argument + ' (e.g. 6616faac453a3064/0000011c--8152ba83cf)')
        else:
            options.segments += parse_segment(argument)
        index += 1

    if not options.route:
        if options.segments or options.skip_mode:
            raise TaskError('the route is missing (e.g. 0000011c--8152ba83cf)')
        options.auto = True
    elif options.skip_mode:
        if options.segments:
            raise TaskError('--skip/--strip-ends does not go with a list of segments')
    elif not options.segments:
        # No segments and no --skip: the default is --strip-ends.
        options.skip = list(STRIP_ENDS)
        options.skip_mode = True
    return options


def videos_dir() -> Path:
    """The shared folder of the comma files, created, as an absolute path (the links in data/ must
    keep working when a segment folder moves to data_scartati/)."""
    folder = os.environ.get('COMMA_VIDEOS_DIR') or str(REPO.parent / 'comma-videos')
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        raise TaskError('cannot create the shared folder ' + folder + ' (COMMA_VIDEOS_DIR)') from None
    if not os.path.isdir(folder):
        raise TaskError('cannot create the shared folder ' + folder + ' (COMMA_VIDEOS_DIR)')
    return Path(os.path.abspath(folder))


def segment_name(route: str, segment: int) -> str:
    """The folder name in data/: 0000011c--8152ba83cf, 3 -> 0000011c8152ba83cf03."""
    return route.replace('-', '') + f'{segment:02d}'


# --- the comma logs ---

def inspect_log(mode: str, *paths: Path) -> str:
    """What helpers/comma_fetch_log.py says of a log (dashcam/normal, seconds, forward/reverse);
    'unknown' when it fails."""
    python = os.environ.get('COMMA_FETCH_PYTHON') or sys.executable
    try:
        result = subprocess.run([python, '-m', 'helpers.comma_fetch_log', mode, *(str(path) for path in paths)],
                                cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True,
                                errors='replace', check=False)
    except OSError:
        return 'unknown'
    if result.returncode != 0:
        return 'unknown'
    return result.stdout.rstrip('\n')


def _pull(comma: Comma, remote_path: str, local_path: Path) -> bool:
    """A copy from the comma that may fail: False (rsync already said why) instead of an error."""
    try:
        comma.pull(remote_path, local_path)
    except TaskError:
        return False
    return True


# --- which segments ---

def parse_listing(listing: str, since: str) -> tuple[list[str], dict[str, int]]:
    """From the stat listing of the comma: the routes with a segment written on or after `since`
    (in listing order) and the lowest segment still on the comma of every route."""
    recent = []
    first_segment = {}
    for line in listing.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        day, folder = parts[0], parts[3].strip()
        route, separator, segment = folder.rpartition('--')
        # Only folders shaped as <route>--<n> with a hex route: the data/ name must be hex.
        if not separator or not ROUTE_RE.fullmatch(route) or not NUMBER_RE.fullmatch(segment):
            continue
        if route not in first_segment or int(segment) < first_segment[route]:
            first_segment[route] = int(segment)
        if day < since:
            continue
        if route not in recent:
            recent.append(route)
    return recent, first_segment


def remote_segments(comma: Comma, route: str) -> list[int]:
    """The segment numbers of a route found on the comma, in order."""
    result = comma.script(ROUTE_SEGMENTS_SCRIPT, route, check=False)
    if result.returncode != 0:
        raise TaskError('the comma does not answer while reading the segments of ' + route)
    numbers = []
    for line in result.stdout.splitlines():
        segment = line.strip().rpartition('--')[2]
        if NUMBER_RE.fullmatch(segment):
            numbers.append(int(segment))
    return sorted(numbers)


def resolve_skip(found: list[int], skip: list) -> list[int]:
    """The numbers of a skip list: "last" is the highest segment found, "last-N" that minus N,
    first/second/second_last the lowest, second lowest and second highest one."""
    first = found[0] if found else None
    second = found[1] if len(found) > 1 else None
    second_last = found[-2] if len(found) > 1 else None
    last = found[-1] if found else None
    numbers = []
    for item in skip:
        if item == 'first':
            numbers.append(first)
        elif item == 'second':
            if second is not None:
                numbers.append(second)
        elif item == 'second_last':
            if second_last is not None:
                numbers.append(second_last)
        elif item == 'last':
            numbers.append(last)
        elif isinstance(item, str) and LAST_MINUS_RE.fullmatch(item):
            numbers.append(last - int(item[len('last-'):]))
        else:
            numbers.append(int(item))
    return numbers


def choose_segments(route: str, found: list[int], skip: list) -> list[int] | None:
    """The segments of `found` not skipped, None (with a message) when the route is not there or
    nothing is left."""
    if not found:
        warn('no segment of ' + route + ' on the comma')
        return None
    skip_numbers = resolve_skip(found, skip)
    segments = [segment for segment in found if segment not in skip_numbers]
    skip_text = ''.join(' ' + str(number) for number in skip_numbers)
    if not segments:
        warn(route + ': skipping' + skip_text + ' leaves no segment')
        return None
    say(route + ' on the comma: ' + ' '.join(str(segment) for segment in found) + '. Skipping:' + skip_text
        + '. Downloading:' + ''.join(' ' + str(segment) for segment in segments))
    return segments


def find_dashcam_routes(comma: Comma, since: str) -> list[str]:
    """Automatic mode: the routes since `since` recorded in dashcam mode (from the qlog of the first
    segment still on the comma)."""
    listing = comma.run(LISTING_COMMAND).stdout
    recent, first_segment = parse_listing(listing, since)
    if not recent:
        say('No route on the comma since ' + since + '.')
        return []
    say('Routes on the comma since ' + since + ':' + ''.join(' ' + route for route in recent))

    dashcam = []
    with tempfile.TemporaryDirectory() as qlog_dir:
        copied = []
        for route in recent:
            qlog = Path(qlog_dir) / (route + '.qlog.zst')
            remote = REMOTE_DIR + '/' + route + '--' + str(first_segment[route]) + '/qlog.zst'
            if _pull(comma, remote, qlog):
                copied.append(route)
            else:
                warn(route + ': qlog of segment ' + str(first_segment[route]) + ' not copied, skipping the route')
        for route in copied:
            qlog = Path(qlog_dir) / (route + '.qlog.zst')
            if not qlog.is_file():
                continue
            state = inspect_log('dashcam', qlog)
            say('  ' + route + ': ' + state)
            if state == 'dashcam':
                dashcam.append(route)
    return dashcam


def route_is_long_enough(comma: Comma, route: str, found: list[int]) -> bool:
    """Automatic mode: the whole trip, from the first to the last qlog of ALL the segments found
    (including the four that are not downloaded), lasts at least 300 s. Both ends are needed: the
    number of folders alone would accept a five-segment route ending before 5 minutes."""
    if not found:
        return False
    with tempfile.TemporaryDirectory() as temp:
        first, last = Path(temp) / 'first.zst', Path(temp) / 'last.zst'
        if not _pull(comma, REMOTE_DIR + '/' + route + '--' + str(found[0]) + '/qlog.zst', first) \
                or not _pull(comma, REMOTE_DIR + '/' + route + '--' + str(found[-1]) + '/qlog.zst', last):
            warn(route + ': first/last qlog not available, trip excluded')
            return False
        seconds = inspect_log('duration', first, last)
    if SECONDS_RE.fullmatch(seconds) and int(seconds.partition('.')[0]) >= MIN_ROUTE_SECONDS:
        say(route + ': full duration ' + seconds + 's (>= ' + str(MIN_ROUTE_SECONDS) + 's), ok')
        return True
    say(route + ': duration ' + seconds + 's (< ' + str(MIN_ROUTE_SECONDS) + 's or unknown), trip excluded')
    return False


def filter_reverse_segments(comma: Comma, route: str, segments: list[int], videos: Path) -> list[int]:
    """Automatic mode: the segments whose full rlog never has reverse gear. A single reverse event
    rejects the segment, and so does an rlog that cannot be read. The rlog is kept in the shared
    folder: the next automatic fetch does not copy it again."""
    remaining = []
    for segment in segments:
        shared = videos / (route + '--' + str(segment))
        logfile = shared / 'rlog.zst'
        if shared.is_symlink() or logfile.is_symlink():
            warn(route + '--' + str(segment) + ': unsafe link in the shared folder, segment excluded')
            continue
        if not (logfile.exists() and logfile.stat().st_size > 0):
            shared.mkdir(parents=True, exist_ok=True)
            if not _pull(comma, REMOTE_DIR + '/' + route + '--' + str(segment) + '/rlog.zst', shared):
                warn(route + '--' + str(segment) + ': rlog not available, segment excluded')
                continue
        state = inspect_log('reverse', logfile)
        if state == 'forward':
            remaining.append(segment)
        elif state == 'reverse':
            say(route + '--' + str(segment) + ': reverse gear detected, segment excluded')
        else:
            warn(route + '--' + str(segment) + ': reverse gear cannot be checked, segment excluded')
    return remaining


# --- the files on disk ---

def move_old_file(old: Path, shared: Path) -> None:
    """Old layout: `old` (in data/) still a regular file downloaded there. It goes to `shared`:
    moved when the shared folder does not have it yet, removed when the shared copy is identical.
    A shared copy that differs is left to the user: stop, nothing deleted. The two being the same
    file (one reaches the other through a link) also stops: they would compare identical and the
    removal would delete the only copy."""
    if old.is_symlink() or not old.exists():
        return
    if not old.is_file():
        raise TaskError(str(old) + ' is neither a file nor a link: check it by hand')
    if not shared.exists():
        if shared.is_symlink():
            # A dangling link takes the name: the move replaces the link, never writes through it.
            shared.unlink()
        shutil.move(str(old), str(shared))
        say('  ' + old.name + ': moved from data/ into the shared folder')
    elif os.path.samefile(old, shared):
        raise TaskError(str(old) + ' and ' + str(shared) + ' are the same file (one reaches the other through a '
                        'link): nothing deleted, check it by hand')
    elif filecmp.cmp(old, shared, shallow=False):
        old.unlink()
        say('  ' + old.name + ': same as the one in the shared folder, removed from data/')
    else:
        raise TaskError(str(old) + ' (' + str(old.stat().st_size) + ' bytes) and ' + str(shared) + ' ('
                        + str(shared.stat().st_size) + ' bytes) differ: nothing deleted, check it by hand')


def link_file(target: Path, link: Path) -> None:
    """Makes `link` (in data/) an absolute link to `target` (in the shared folder)."""
    if link.is_symlink() and os.readlink(link) == str(target):
        return
    if link.is_symlink():
        link.unlink()
    try:
        os.symlink(str(target), link)
    except OSError as exc:
        raise TaskError('cannot create the link ' + str(link) + ': ' + exc.strerror) from None


def check_real_folders(dst: Path, shared: Path) -> None:
    """A segment folder that is a link would make data/<name>/<file> and the shared file the same
    file (the localizer outputs would also end up in the shared folder): stop."""
    if dst.is_symlink():
        raise TaskError(str(dst) + ' is a link to a folder: data/ needs a real folder holding the links to the '
                        'files. Nothing deleted, check it by hand')
    if shared.is_symlink():
        raise TaskError(str(shared) + ' is a link to a folder: the shared folder needs a real folder. '
                        'Nothing deleted, check it by hand')


def run_localizer(name: str) -> None:
    environment = dict(os.environ)
    environment['PYTHONUNBUFFERED'] = '1'
    code = subprocess.run([sys.executable, '-m', 'localizer.run', '--segment', name], cwd=REPO, env=environment,
                          stdin=subprocess.DEVNULL, check=False).returncode
    if code != 0:
        raise TaskError('localization of ' + name + ' failed (code ' + str(code) + ')')


def download_segments(comma: Comma, route: str, segments: list[int], videos: Path, localizer: bool) -> None:
    """Copies `segments` of `route` into the shared folder, links them into data/, then runs the
    localizer on each when asked."""
    say('PROGRESS 0/' + str(len(segments)))
    for done, segment in enumerate(segments, start=1):
        src = REMOTE_DIR + '/' + route + '--' + str(segment)
        shared = videos / (route + '--' + str(segment))
        name = segment_name(route, segment)
        dst = DATA_DIR / name
        # Discarded by the localizer (tasks.localize): not again.
        if (DISCARD_DIR / name).is_dir():
            say('== ' + route + '--' + str(segment) + ': discarded by the localizer (data_scartati/' + name
                + '), skipping')
            say('PROGRESS ' + str(done) + '/' + str(len(segments)))
            continue
        say('== ' + route + '--' + str(segment) + ' -> ' + str(shared) + ' (links in ' + str(dst) + ')')

        for file_name in FILES:
            if comma.script(FILE_EXISTS_SCRIPT, src + '/' + file_name, check=False).returncode != 0:
                raise TaskError(src + '/' + file_name + ' does not exist on the comma')

        check_real_folders(dst, shared)
        shared.mkdir(parents=True, exist_ok=True)
        dst.mkdir(parents=True, exist_ok=True)
        for file_name in FILES:
            move_old_file(dst / file_name, shared / file_name)
            comma.pull(src + '/' + file_name, shared, progress=True)
            link_file(shared / file_name, dst / file_name)

        if localizer:
            run_localizer(name)
        say('PROGRESS ' + str(done) + '/' + str(len(segments)))


def fetch(options: Options) -> int:
    videos = videos_dir()
    comma = Comma(comma_host(options.host))
    comma.require()

    if options.auto:
        routes = find_dashcam_routes(comma, options.since)
        if not routes:
            say('No route in dashcam mode to download.')
            return 0
        say('In dashcam mode:' + ''.join(' ' + route for route in routes))
        for route in routes:
            found = remote_segments(comma, route)
            segments = choose_segments(route, found, list(STRIP_ENDS))
            if segments and route_is_long_enough(comma, route, found):
                segments = filter_reverse_segments(comma, route, segments, videos)
                if segments:
                    download_segments(comma, route, segments, videos, options.localizer)
                    continue
            say(route + ': no eligible segment to download')
    else:
        segments = options.segments
        # --skip / --strip-ends: read the segments on the comma, then drop the skipped ones.
        if options.skip_mode:
            segments = choose_segments(options.route, remote_segments(comma, options.route), options.skip)
            if segments is None:
                return 1
        download_segments(comma, options.route, segments, videos, options.localizer)

    say('Done.')
    return 0


def main() -> int:
    options = parse_args(sys.argv[1:])
    if options.help:
        print(__doc__)
        return 0
    load_env()
    try:
        return fetch(options)
    except OSError as exc:
        raise TaskError('disk error: ' + str(exc)) from exc


if __name__ == '__main__':
    run_main(main)
