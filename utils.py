import json
from collections import OrderedDict

## json file IO
def read_json(fname):
    with open(fname, 'rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with open(fname, 'wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def write_json_no_indent(content, fname):
    with open(fname, 'wt') as handle:
        json.dump(content, handle, sort_keys=False)