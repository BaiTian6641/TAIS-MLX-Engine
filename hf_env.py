"""Hugging Face endpoint, credential and cache configuration.

Every entry point that touches the Hub imports this first, because
``huggingface_hub`` reads ``HF_ENDPOINT`` when it builds request URLs and
reading it late is the same as not setting it.

Three things are configurable, each from a flag, then from a ``K2MLX_``
variable, then from the conventional Hugging Face variable:

===========================  =================================  ==========================
Setting                      Preferred                          Also read
===========================  =================================  ==========================
Mirror endpoint              ``K2MLX_HF_ENDPOINT``              ``HF_ENDPOINT``
Access token                 ``K2MLX_HF_TOKEN``                 ``HF_TOKEN``
Cache directory              ``K2MLX_HF_HOME``                  ``HF_HOME``
===========================  =================================  ==========================

A mirror is any host that serves the Hub API, for example
``https://hf-mirror.com``. The value is normalised to a bare scheme and host so
that a trailing slash or a copied URL path cannot produce a double slash in
every request.
"""
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


def _first(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip().rstrip('/')
    return None


def normalise_endpoint(value):
    """Return ``scheme://host[:port]`` for a mirror URL, or raise on nonsense."""
    if value is None:
        return None
    parts = urlsplit(value if '://' in value else f'https://{value}')
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        raise ValueError(f'HF endpoint must be an http(s) URL, got {value!r}')
    # Only scheme and host survive: the API lives at the host root, and a path
    # copied from a model page would otherwise be pasted onto every request.
    return f'{parts.scheme}://{parts.netloc}'


def configure(*, endpoint=None, token=None, home=None, offline=False, root=None,
              verbose=False):
    """Apply Hugging Face settings to the process environment.

    Returns the effective configuration, which ``doctor`` prints and which
    callers can log so a run records where its weights came from.
    """
    root = Path(root or ROOT)
    endpoint = normalise_endpoint(endpoint) or normalise_endpoint(_first('K2MLX_HF_ENDPOINT', 'HF_ENDPOINT'))
    token = token or _first('K2MLX_HF_TOKEN', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')
    home = home or _first('K2MLX_HF_HOME', 'HF_HOME') or str(root / 'hf-cache')

    os.environ['HF_HOME'] = str(home)
    if endpoint:
        os.environ['HF_ENDPOINT'] = endpoint
    if token:
        os.environ['HF_TOKEN'] = token
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    applied = {
        'endpoint': endpoint or os.environ.get('HF_ENDPOINT') or 'https://huggingface.co',
        'token': 'set' if (token or os.environ.get('HF_TOKEN')) else 'unset',
        'home': str(home),
        'offline': bool(offline or os.environ.get('HF_HUB_OFFLINE')),
    }
    if verbose:
        print(f"hugging face: endpoint={applied['endpoint']} token={applied['token']} "
              f"cache={applied['home']}")
    return applied


def add_arguments(parser):
    """Attach the Hub options to an argparse parser."""
    group = parser.add_argument_group('hugging face')
    group.add_argument('--hf-endpoint', metavar='URL',
                       help='Hub mirror or endpoint, e.g. https://hf-mirror.com '
                            '(default: $K2MLX_HF_ENDPOINT, then $HF_ENDPOINT)')
    group.add_argument('--hf-token', metavar='TOKEN',
                       help='access token for gated or private repositories '
                            '(default: $K2MLX_HF_TOKEN, then $HF_TOKEN)')
    group.add_argument('--hf-home', metavar='DIR',
                       help='where downloaded files are cached (default: <repo>/hf-cache)')
    group.add_argument('--hf-offline', action='store_true',
                       help='use only what is already cached; never call the network')
    return parser


def apply_from_args(args, **kwargs):
    """Configure from parsed ``add_arguments`` options."""
    return configure(endpoint=getattr(args, 'hf_endpoint', None),
                     token=getattr(args, 'hf_token', None),
                     home=getattr(args, 'hf_home', None),
                     offline=getattr(args, 'hf_offline', False), **kwargs)


def endpoint():
    """The endpoint to pass to a Hub call, or ``None`` for the default."""
    value = os.environ.get('HF_ENDPOINT')
    return None if not value or value.rstrip('/') == 'https://huggingface.co' else value


def token():
    return os.environ.get('HF_TOKEN') or None


def hub_kwargs():
    """Keyword arguments that route a ``huggingface_hub`` call through the mirror."""
    kwargs = {}
    if endpoint():
        kwargs['endpoint'] = endpoint()
    if token():
        kwargs['token'] = token()
    return kwargs
