import copy as cp

class PyQueue:
    def __init__(self, size=20):
        self.queue = []
        self.size = size
        self.end = -1


    def SetSize(self, size):
        self.size = size


    def In(self, element):
        if self.end < self.size - 1:
            self.queue.append(element)
            self.end += 1
        else:
            raise QueueException('PyQueueFull')


    def Out(self):
        if self.end == -1:
            raise QueueException('PyQueueEmpty')
        else:
            element = self.queue[0]
            self.queue = self.queue[1:]
            self.end -= 1
            return element


    def End(self):
        return self.end


    def clear(self):
        self.queue = []
        self.end = -1

    def isEmpty(self):
        return len(self.queue)==0


class QueueException(Exception):
    def __init__(self, data):
        self.data = data
    def __str__(self):
        return self.data

if __name__ == '__main__':
    q1 = PyQueue(1)
    q1.In(1)
    q2 = cp.copy(q1)
    print q2.Out()
    print q1.isEmpty()
    print q1.Out()
    print q1.isEmpty()

