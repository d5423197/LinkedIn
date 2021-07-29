class Solution(object):
    def numIslands(self, grid):
        count = 0
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    'we found 1 island'
                    count+=1
                    'now we need to mark all the adjacent lands'
                    self.dfs(i, j, m, n, grid)
        return count
    def dfs(self, i, j, m, n, grid):
        'if we find a cell which is not 1, meaning we either find a visited cell or a 0, so we have to return'
        if grid[i][j] != '1':
            return
        'mark current cell as visited'
        grid[i][j] = 'visited'
        for i_inc, j_inc in zip([1, 0, -1, 0], [0, 1, 0, -1]):
            'now we just need to move up, bottom, right and left'
            new_i = i + i_inc
            new_j = j + j_inc
            'we need to make sure the index is not out of bound'
            if 0 <= new_i <= m - 1 and 0 <= new_j <= n - 1:
                self.dfs(new_i, new_j, m, n, grid)