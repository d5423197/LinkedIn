class MinHeap:
    def __init__(self, input = []):
        self.heapList = [0]
        self.currentSize = 0
        for i in input:
            self.heapList.append(i)
            self.currentSize+=1
            self.go_up(self.currentSize)

    def go_up(self, i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                self.heapList[i], self.heapList[i // 2] = self.heapList[i // 2], self.heapList[i]
            i = i // 2 # here you only care about current node's mother

    def add(self, i):
        self.heapList.append(i)
        self.currentSize+=1
        self.go_up(self.currentSize)

    def go_down(self, i):
        while i * 2 <= self.currentSize:
            min_child = self.minChild(i)
            if self.heapList[i] > self.heapList[min_child]:
                self.heapList[i], self.heapList[min_child] = self.heapList[min_child], self.heapList[i]
            i = i * 2

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2 + 1] < self.heapList[i * 2]:
                return i * 2 + 1
            else:
                return i * 2

    def deleteMin(self):
        minimum = self.heapList[1]
        self.heapList[1] = self.heapList[-1]
        self.currentSize-=1
        self.heapList.pop()
        self.go_down(1)
        return minimum

    def __repr__(self):
        return str(self.heapList)
test = [9, 6, 5, 2, 3, 10, 20, -1]
obj = MinHeap(input = test)
print(obj)
obj.add(-50)
print(obj)
