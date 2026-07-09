from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.store: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self.store:
            return None
        return self.store[key]  # BUG: does not mark key as recently used

    def put(self, key, value):
        if key in self.store:
            self.store[key] = value
            return
        if len(self.store) >= self.capacity:
            self.store.popitem(last=False)
        self.store[key] = value
