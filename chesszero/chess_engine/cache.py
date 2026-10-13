"""Bounded caches for model-independent chess-rule results."""

from collections import OrderedDict
from collections.abc import MutableMapping


class LRUCache(MutableMapping):
    def __init__(self, max_size=10_000):
        if max_size < 1:
            raise ValueError("Cache size must be positive")
        self.max_size = max_size
        self._entries = OrderedDict()

    def __getitem__(self, key):
        value = self._entries[key]
        self._entries.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        self._entries[key] = value
        self._entries.move_to_end(key)
        if len(self._entries) > self.max_size:
            self._entries.popitem(last=False)

    def __delitem__(self, key):
        del self._entries[key]

    def __iter__(self):
        # Reading values refreshes LRU order, including during items() iteration.
        return iter(tuple(self._entries))

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()
