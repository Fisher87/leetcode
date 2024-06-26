#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2023 Fisher. All rights reserved.
#   
#   文件名称：algorithm.py
#   创 建 者：YuLianghua
#   创建日期：2023年02月11日
#   描    述：
#
#================================================================

class MaxProfit:
    # https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/?favorite=2cktkvj
    def maxProfit(self, prices):
        n = len(prices)
        dp = [[0]*3 for _ in range(n)]
        for i in range(n):
            if i==1:
                dp[i][0] = -prices[i]
            else:
                dp[i][0] = max(dp[i-1][0], dp[i-1][2]-prices[i])
                dp[i][1] = dp[i-1][0]+prices[i]
                dp[i][2] = max(dp[i-1][2], dp[i-1][1])

        return max(dp[n-1][1], dp[n-1][2])

class canPartition:
    # 分割等和子集
    def canpartition(self, nums):
        if len(nums)<2:
            return False
        sums = sum(nums)
        if sums&1:
            return False
        nums.sort()

        target = sums // 2
        if nums[-1] > target:
            return False

        n = len(nums)
        # 转化为判断从nums中选取数字组合，和为target 是否存在;
        dp = [[False]*(target+1) for _ in range(n)]
        for i in range(n):
            dp[i][0] = True

        dp[0][nums[0]] = True
        for i in range(1, n):
            num = nums[i]
            for j in range(1, target+1):
                if j>=num:
                    dp[i][j] = dp[i-1][j] | dp[i-1][j-num]
                else:
                    dp[i][j] = dp[i-1][j]

        return dp[n-1][target]

class FindAnagrams(object):
    def findAnagrams(self, s, p):
        s_len = len(s)
        p_len = len(p)
        if s_len<p_len:
            return []
        result = []
        s_count = [0]*26
        p_count = [0]*26
        for i in range(p_len):
            s_count[ord(s[i])-ord('a')] += 1
            p_count[ord(p[i])-ord('a')] += 1
        if s_count == p_count:
            result.append(0)
        for i in range(1, s_len-p_len):
            s_count[ord(s[i-1])-ord('a')] -= 1
            s_count[ord(s[i+p_len])-ord('a')] += 1
            if s_count == p_count:
                result.append(i)

        return result

class BitCount:
    def bitcount(self, n):
        result = [0]*(n+1)
        for i in range(1, n+1):
            if i%2==0:
                result[i] = result[i>>1] 
            else:
                result[i] = result[i-1] + 1
        return result

class EditDistance:
    def edit_dist(self, word1, word2):
        m, n = len(word1), len(word2)
        dp = [ [0]*(n+1) for _ in range(m+1)]

        for i in range(1, m+1):
            dp[i][0] = i
        for j in range(1, n+1):
            dp[0][j] = j
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[m][n]
    
class zigzagLevelOrder:
    # 锯齿型层次遍历
    def zigzagLevelOrder(self, root):
        rflag = True
        stack = []
        stack.append(root)
        result = []
        while stack:
            size = len(stack)
            res = []
            for _ in range(size):
                node = stack.pop(0)
                res.append(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
            if not rflag:
                res = res[::-1]
            result.append(res)
            rflag = not rflag
        return result

import heapq
class FindKthLargest:
    # 返回数组中第 K 大的元素
    def findkth_largest(self, nums, k):
        heap = []
        for i in range(k):
            heapq.heappush(heap, nums[i])

        for n in nums[k:]:
            kth = heapq.heappop(heap)
            if n > kth:
                heapq.heappush(heap, n)
            else:
                heapq.heappush(heap, kth)

        return heapq.heappop(heap)


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left= None
        self.right=None
class LowestCommonAncestor:
    # 最近公共祖先
    def lowest_common_ancestor(self, root, p, q):
        self.ans = None
        def dfs(root, p, q):
            if not root:
                return 
            lson = dfs(root.left, p, q)
            rson = dfs(root.right, p, q)
            if (lson and rson) or ((root.val==p.val or root.val==q.val) and (lson or rson)):
                self.ans = root
            return lson or rson or root.val==p.val or root.val==q.val
        dfs(root, p, q)

        return self.ans

    def lowestCommonAncestor(self, root, p, q):
        if not root or root == p or root == q:return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left: return right
        if not right: return left
        return root

import bisect

class LengthLIS:
    # 最长上升子序列
    def lengthLIS(self, nums):
        # 使用dp
        n = len(nums)
        dp = [1]*n
        for i in range(n):
            for j in range(i):
                if nums[j]<nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)

    def lengthLIS2(self, nums):
        # 二分查找
        ans = []
        for n in nums:
            if not ans or n>ans[-1]:
                ans.append(n)
            else:
                # 查找替换
                loc = bisect.bisect_left(ans, n)
                ans[loc] = n

        return len(ans)

class Search:
    # 搜索旋转排序数组
    def search(self, nums, target):
        l, r = 0, len(nums)-1
        while l<=r:
            mid = (l+r)//2
            if nums[mid]==target:
                return mid
            if nums[0]<=nums[mid]:
                if nums[0]<=target<nums[mid]:
                    r = mid-1
                else:
                    l = mid+1
            else:
                if nums[mid]<target<=nums[-1]:
                    l = mid+1
                else:
                    r = mid-1

        return -1

class LengthofLongestSubstring:
    # 最长无重复字符子串
    def lengthofLongestSubstring(self, s):
        # 使用窗口
        window = []
        ret = 0
        for c in s:
            while c in set(window):
                window.pop(0)
            window.append(c)
            ret = max(ret, len(window))
        return ret
    
    def lengthofLongestSubstring2(self, s):
        start, maxlen = -1, 0
        cpos = dict()
        for i, c in enumerate(s):
            if c in cpos and cpos[c]>start:
                start = cpos[c]
                cpos[c] = i
            else:
                cpos[c] = i
                maxlen = max(maxlen, i-start)

        return maxlen

class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None

class MergeKLists:
    # 合并K个升序链表 : https://leetcode.cn/problems/merge-k-sorted-lists/description/
    def mergeKLists(self, lists):
        def merge(lists, l, r):
            if l==r:
                return lists[l]
            if l>r:
                return None
            mid = (l+r)//2
            return self.mergeTwolists(merge(lists, l, mid), merge(lists, mid+1, r))
        merge(lists, 0, len(lists)-1)

    def mergeTwolists(self, left, right):
        dummy = ListNode()
        head = dummy
        cura , curb = left, right
        while cura and curb:
            if cura.val < curb.val:
                head.next = cura
                cura = cura.next
            else:
                head.next = curb
                curb = curb.next
            head = head.next

        if cura:
            head.next = cura
        if curb:
            head.next = curb

        return dummy.next

    def mergeKLists2(self, lists):
        import heapq
        dummy = ListNode()
        cur = dummy
        pq = []
        n = len(lists)

        for i in range(n):
            if lists[i] is None:continue
            heapq.heappush(pq, (lists[i].val, i))

        while pq:
            _, i = heapq.heappop(pq)
            cur.next = lists[i]
            cur = cur.next
            lists[i] = lists[i].next
            if lists[i] is not None:
                heapq.heappush(pq, (lists[i].val, i))

        return dummy.next

        
class FindMedianSortedArrays:
    # 从两个排序数组中找出中位数
    def findMediansortedArrays(self, nums1, nums2):
        m, n = len(nums1), len(nums2)
        if (m+n) % 2: # 奇数个
            median_k = (m+n)//2 +1
            return self.get_k(nums1, nums2, median_k)
        else:
            median_l, median_r = (m+n)//2, (m+n)//2+1
            return (self.get_k(nums1, nums2, median_l) + self.get_k(nums1, nums2, median_r))*0.5
        
    def get_k(self, nums1, nums2, k):
        index1, index2 = 0, 0
        while True:
            # 结束分三种情况：1. nums1已经匹配结束; 2. nums2已经匹配结束; 3. k已经结束;
            if index1==len(nums1):
                return nums2[index2+k-1]
            if index2==len(nums2):
                return nums1[index1+k-1]
            if k==1:
                return min(nums1[index1], nums2[index2])

            _index1 = min(index1+k//2 -1, len(nums1)-1)
            _index2 = min(index2+k//2 -1, len(nums2)-1)
            if nums1[_index1] <= nums2[_index2]:
                k = k-(_index1-index1+1)   # 因为_index1 是由 min 获取到的，所以相差不一定是 k//2;
                index1 = _index1+1
            else:
                k = k-(_index2-index2+1)
                index2 = _index2+1

class SpiralOrder:
    # 顺时针打印矩阵
    def spiralOrder(self, matrix):
        if not matrix:
            return []
        m, n = len(matrix), len(matrix[0])
        l, r, t, b = 0, n-1, 0, m-1
        ans = []
        total = m*n
        count = 0
        while count<total:
            if l>r: break
            for i in range(l, r+1):
                ans.append(matrix[t][i])
                count += 1
            t += 1
            if t>b: break
            for i in range(t, b+1):
                ans.append(matrix[i][r])
                count += 1
            r -= 1
            if r<l: break
            for i in range(r, l-1, -1):
                ans.append(matrix[b][i])
                count += 1
            b -= 1
            if b<t:break
            for i in range(b, t-1, -1):
                ans.append(matrix[i][l])
                count += 1
            l += 1

        return ans

            
class MaxAreaOfIsland:
    # 请最大岛屿
    def maxAreaOfIsland(self, grid):
        m, n = len(grid), len(grid[0])
        def valid(i, j):
            if 0<=i<m and 0<=j<n:
                return True
            return False

        def dfs(i, j):
            if not valid(i, j) or grid[i][j]!=1:
                return 0
            _ans = 1
            grid[i][j] = 0
            for dx, dy in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                _i, _j = i+dx, j+dy
                _ans += dfs(_i, _j)
            return _ans

        ans = -1
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                   ans = max(ans, dfs(i, j))
        return ans
    
class LevelOrder:
    # 返回二叉树层序遍历
    def levelorder(self, root):
        # bfs
        if not root:
            return []
        stack = []
        ans = []
        stack.append(root)
        while stack:
            size = len(stack)
            for _ in range(size):
                node = stack.pop(0)
                ans.append(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

        return ans

class SearchMatrix:
    # 搜索二维矩阵
    def searchMatrix(self, matrix, target):
        i, j = 0, len(matrix[0])-1
        while i<len(matrix)-1 and j>=0:
            if matrix[i][j] == target:
                return True
            if matrix[i][j]<target:
                i += 1
            else:
                j -= 1
        return False
                
class MinsubArrayLen:
    # 和>=target的最短子数组
    def minsubArraylen(self, target, nums):
        n = len(nums)
        min_len = n+1
        start, end = 0, n-1
        while start<=end:
            if sum(nums[start:end+1])<target:
                break
            else:
                min_len = min(min_len, end-start+1)
                if nums[start]<=nums[end]:   # 这里会有问题
                    start += 1
                else:
                    end -= 1
        if min_len>n:
            return 0
        return min_len

    def minsubArraylen(self, target, nums):
        n = len(nums)
        start = 0
        total = 0
        min_len = n+1
        for i in range(n):
            total += nums[i]
            if total>=target:
                min_len = min(min_len, i-start+1)
                total -= nums[start]
                start += 1
        if min_len>n:
            return 0
        return min_len

class IsSymmetric:
    # 对称二叉树
    def isSymmetric(self, root):
        if not root:
            return True
        def helper(a, b):
            if not a and not b:
                return True
            if not a or not b or (a.val!=b.val):
                return False

            return helper(a.left, b.right) and helper(a.right, b.left)

        return helper(root.left, root.right)

class IsBalanced:
    # 是否为平衡二叉树(左右子树高度差不能超过1)
    def isBalanced(self, root):
        if not root:
            return True
        l_depth = self.depth(root.left) 
        r_depth = self.depth(root.right)
        return abs(l_depth-r_depth)<=1 and self.isBalanced(root.left) and self.isBalanced(root.right)

    def depth(self, root):
        if not root:
            return 0
        return 1+max(self.depth(root.left), self.depth(root.right))

class MaximalRectangle:
    # 最大矩形面积
    # 分两步：1. 计算出每一行的柱状图;
    #         2. 计算每一行柱状图最大矩形面积；
    def maximalRectangle(self, matrix):
        matrix = [list(map(int, x)) for x in matrix]
        hists = self.get_hists(matrix)
        marea = 0
        for hist in hists:
            marea = max(marea, self.max_area(hist))

        return marea

    def get_hists(self, matrix):
        m, n = len(matrix), len(matrix[0])
        hists = [matrix[0]]
        for i in range(1,m):
            hist = []
            for j in range(n):
                if matrix[i][j] == 0:
                    hist.append(0)
                else:
                    hist.append(hists[-1][j]+1)
            hists.append(hist)
        return hists

    def max_area(self, hist):
        # 求柱状图最大矩形面积，使用单调栈获取当前位置左右第一个较小的数的位置
        n = len(hist)
        stack = []
        left = [-1]*n
        for i in range(n):
            while stack and hist[stack[-1]]>=hist[i]:
                stack.pop()
            if stack:
                left[i] = stack[-1]
            stack.append(i)

        stack = []
        right = [n]*n
        for i in range(n-1, -1, -1):
            while stack and hist[stack[-1]]>=hist[i]:
                stack.pop()
            if stack:
                right[i] = stack[-1]
            stack.append(i)

        _max = 0
        for i in range(n):
            _max = max(_max, hist[i]*(right[i]-left[i]-1))

        return _max

class FindDuplicate:
    # 寻找重复数
    def findDuplicate(self, nums):
        slow, fast = 0, 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow==fast:
                break
        fast = 0
        while slow!=fast:
            slow = nums[slow]
            fast = nums[fast]

        return fast

class FindDuplicates:
    # 数组中的重复数据
    def findduplicates(self, nums):
        for i in range(len(nums)):
            while nums[i] != nums[nums[i]-1]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        return [num for i,num in enumerate(nums) if num!=(i+1)]

    def findduplicates(self, nums):
        ans = []
        for num in nums:
            x = abs(num)
            if nums[x-1] > 0:
                nums[x-1] = -num[x-1]
            else:
                ans.append(x)
        return ans

class MinPathSum:
    # 最小路径和
    def minPathSum(self, grid):
        m, n = len(grid), len(grid[0])
        dp = [ [0]*n for _ in range(m) ]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[j][0]

        for i in range(m):
            for j in range(n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

        return dp[-1][-1]

class longestPalindrome:
    # 最长回文子串
    def longestPalindrome(self, s):
        n = len(s)
        if n<=1:
            return s
        
        maxlen, max_s = 0, ''
        for i in range(n):
            left, right = i-1, i
            while right<n and s[right]==s[i]:
                right += 1

            while left>=0 and right<n and s[right]==s[left]:
                left -= 1
                right+= 1

            _maxlen = right-(left+1)
            if _maxlen>maxlen:
                maxlen = _maxlen
                max_s = s[left+1:right]

        return max_s

class Permute:
    # 全排列组合
    def permute(self, nums):
        self.ans = []
        size = len(nums)
        visited = [False] * size
        path = []
        self.trace(nums, size, visited, 0, path)

    def trace(self, nums, size, visited, depth, path):
        if depth==size:
            self.ans.append(path[:])
            return 

        for i in range(size):
            if visited[i]:
                continue
            path.append(nums[i])
            visited[i] = True
            self.trace(nums, size, visited, depth+1, path)
            visited[i] = False
            path.pop()

class ThreeSum:
    def threeSum(self, nums):
        n = len(nums)
        if n<3:
            return []
        nums.sort()
        if nums[0]>0:
            return []

        ans = []
        for i in range(n):
            if nums[i]>0:
                break
            if i>0 and nums[i]==nums[i-1]:
                continue
            l, r = i+1, n-1
            while l<r:
                if (nums[i]+nums[l]+nums[r])==0:
                    ans.append(nums[i], nums[l], nums[r])
                    while l<r and nums[l+1]==nums[l]:
                        l += 1
                    while l<r and nums[r-1]==nums[r]:
                        r = r-1
                elif (nums[i]+nums[l]+nums[r])>0:
                    r = r-1
                else:
                    l = l+1
        return ans


class MaxProfit:
    # 买卖股票的最佳时机
    # 思路：找出最小值，然后遍历找出max
    def maxProfit(self, prices):
        min_price = prices[0]
        max_ = 0
        for price in prices[1:]:
            if price<min_price:
                min_price = price
            max_ = max(max_, price-min_price)
        return max_

class IsSubTree:
    # 另一棵树的子树
    def isSubtree(self, root, subroot):
        def sametree(root, subroot):
            if not root and not subroot:
                return True
            if (not root and subroot) or (root and not subroot) or (root.val!=subroot.val):
                return False
            return sametree(root.left, subroot.left) and sametree(root.right, subroot.right)

        return sametree(root, subroot) or sametree(root.left, subroot) or sametree(root.right, subroot)

from collections import defaultdict
class TwoSum:
    # 两数之和等于target， 返回下标
    def twoSum(self, nums, target):
        n_inxs = defaultdict(set)
        for i,n in enumerate(nums):
            n_inx[n].add(i)

        for i,n in enumerate(nums):
            _t = target-n
            if _t in n_inxs:
                s = n_inxs[_t]
                if i in s:
                    s.remove(i)
                if s:
                    return [i, list(s)[0]]

class BuildTree:
    # 由前序中序遍历构造树
    def buildTree(self, preorder, inorder):
        if not preorder:
            return None

        root = TreeNode(val=preorder[0])
        stack = [root]
        index = 0
        for i in range(1, len(preorder)):
            node = TreeNode(val=preorder[i])
            if inorder[index]!=stack[-1].val:
                stack[-1].left=node
                stack.append(node)
            else:
                x = stack[-1]
                while stack and stack[-1].val==inorder[index]:
                    x = stack.pop()
                    index += 1
                x.right = node
                stack.append(node)

        return root

class ReverseList:
    # 反转链表
    def reverseList(self, head):
        if not head:
            return head
        cur, pre = head, None
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt

        return pre

class Merge:
    # 合并两个有序数组 nums2 -> nums1中
    def merge(self, nums1, m, nums2, n):
        inx = m+n-1
        i, j = m-1, n-1
        while i>=0 and j>=0:
            if nums1[i]<nums2[j]:
                nums1[inx] = nums2[j]
                j -= 1
            else:
                nums1[inx] = nums1[i]
                i -= 1
            inx -= 1
        if j>=0:
            nums1[:inx+1] = nums2[:j+1]
        return nums1

class MaxProduct:
    # 乘积最大子数组
    def maxProduct(self, nums):
        n = len(nums)
        maxdp, mindp = [0]*n, [0]*n
        maxdp[0] = mindp[0] = nums[0]
        for i in range(1, n):
            maxdp[i] = max(maxdp[i-1]*nums[i], mindp[i-1]*nums[i], nums[i])
            mindp[i] = max(maxdp[i-1]*nums[i], mindp[i-1]*nums[i], nums[i])

        return max(maxdp)

class Rotate:
    # 旋转图片
    def rotate(self, matrix):
        n = len(matrix)
        # 左右翻转
        for i in range(n):
            for j in range(n//2):
                matrix[i][j], matrix[i][n-j-1] = matrix[i][n-j-1], matrix[i][j]

        # 次对角翻转
        for i in range(n-1):
            for j in range(n-1):
                if (i+j)>=n-1:break
                matrix[i][j], matrix[n-j-1][n-i-1] = matrix[n-j-1][n-i-1], matrix[i][j]

        return matrix

class TreeToDoublyList:
    # 将二叉搜索树变成双向链表
    def treeToDoublyList(self, root):
        self.head, self.pre = None, None
        self.inorder(root)

        self.head.left = self.pre
        self.pre.right = self.head

    def inorder(self, root):
        if not root:
            return
        self.inorder(root.left)
        if not self.pre:
            self.head = root

        else:
            self.pre.right = root
            root.left = self.pre
        self.pre = root
        self.inorder(root.right)

from collections import deque
class MaxSlidingWindow:
    # 滑动窗口最大值
    # 暴力求解太慢
    def maxSlidingWindow(self, nums, k):
        n = len(nums)
        if n<=k:
            return max(nums)
        
        dq = deque()
        for i in range(k):
            while dq and nums[i]>=nums[ dp[-1] ]:
                dq.pop()
            dq.append(i)
        ans = [nums[dq[0]]]

        for i in range(k,n):
            while dq and nums[i]>=nums[ dq[-1] ]:
                dq.pop()
            dq.append(i)
            while dq and dq[0] <= (i-k):
                dq.popleft()
            ans.append( nums[dq[0]] )
        return ans

    def maxslidingwindow(self, nums, k):
        ans = []
        dq = deque()
        for i in range(len(nums)):
            while dq and nums[i]>=nums[dq[-1]]:
                dq.pop()
            dq.append(i)
            while dq and (i-dq[0])>=k:
                dq.popleft()
            if i>=k-1:
                ans.append(nums[dq[0]])
        return ans

class ReorderList:
    # 重排链表
    def reorderList(self, head):
        if not head:
            return
        fast, slow = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        l1 = head
        l2 = slow.next
        slow.next = None     # 重要：表示l1 的tail， 要不然在merger的时候会有问题

        l2 = self.reverse(l2)
        self.merge(l1, l2)

    def reverse(self, root):
        cur, pre = root, None
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

    def merge(self, l1, l2):
        while l1 and l2:
            l1_tmp = l1.next
            l2_tmp = l2.next

            l1.next = l2
            l1 = l1_tmp

            l2.next = l1
            l2 = l2_tmp

class Bit:
    # 构建树状数组
    def __init__(self, n):
        self.tree = [0] * (n+1)
        self.n = n

    @staticmethod
    def bit(x):
        return x & (-x)

    def query(self, _id):
        ans = 0
        while _id > 0:
            ans += self.tree[_id]
            _id -= self.bit(_id)
        return ans

    def update(self, _id):
        while _id < self.n:
            self.tree[_id] += 1
            _id += self.bit(_id)

class ReversePairs:
    # 数组中的逆序对
    # 使用树状数组 / 归并排序
    def reversePairs(self, nums):
        self.ans = 0
        n = len(nums)
        bit = Bit(n)
        _nums = sorted(nums)
        for i in range(n-1, -1, -1):
            _id = self.get_id(_nums, nums[i])
            self.ans += bit.query(_id-1)
            bit.update(_id)
        return self.ans

    def get_id(self, nums, x):
        return bisect.bisect_left(nums, x)+1

    def reversePairs_by_merge(self, nums):
        self.ans = 0
        def partion(nums, left, right):
            if left>=right:
                return nums[left:right+1]
            mid = left + (right-left)//2
            l = partion(nums, left, mid)
            r = partion(nums, mid+1, r)
            return merge(l, r)

        def merge(left, right):
            merge_res = []
            l_size, r_size = len(left), len(right)
            i, j = 0, 0
            while i<l_size and j<r_size:
                if left[i]<=right[j]:
                    merge_res.append(left[i])
                    i += 1
                else:
                    # 说明左边大于右边，为逆序对
                    self.ans += (l_size-i)
                    merge_res.append(right[j])
                    j += 1

            if j<r_size:
                merge_res += right[j:]
            if i<l_size:
                merge_res += left[i:]
            return merge_res

        partion(nums, 0, len(nums)-1)
        return self.ans

class ProductExceptSelf:
    # 除自身以外数组的乘积
    def productExceptSelf(self, nums):
        n = len(nums)
        ans = [1]*n

        left, right = 1, 1
        for i in range(n):
            ans[i] *= left
            left *= nums[i]
            ans[n-1-i] *= right
            right *= nums[n-1-i]

        return ans

class ReverseKGroup:
    # K 个一组翻转链表
    # 1 -> 2 -> 3 -> 4 -> 5 : 2 -> 1 -> 4 -> 3 -> 5
    def reverseKGroup(self, head, k):
        dummynode = ListNode()
        dummynode.next = head

        pre = dummynode
        while head:
            tail = pre
            for i in range(k):
                tail = tail.next
                if not tail:   # 长度小于k
                    return dummynode.next
            nxt = tail.next
            head, tail = self.reverse(head, tail)
            pre.next = head
            tail.next = nxt
            pre = tail
            head = tail.next

        return dummynode.next

    def reverse(self, head, tail):
        pre, cur = head, head
        while pre!=tail:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return tail, head

class Sort:
    def quick_sort(self, nums):
        '''快排
        '''
        def partion(nums, left, right):
            if left>=right:
                return 
            piv = random.randint(left, right)
            nums[piv], nums[right] = nums[right], nums[piv]
            i = left
            for j in range(left, right):
                if nums[j]<nums[right]:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
            nums[i], nums[right] = nums[right], nums[i]
            partion(nums, left, i-1)
            partion(nums, i+1, right)
        partion(nums, 0, len(nums)-1)

        return nums

    def merge_sort(self, nums):
        def mergesort(nums, l, r):
            if l == r:
                return
            mid = (l + r) // 2
            mergesort(nums, l, mid)
            mergesort(nums, mid + 1, r)
            tmp = []
            i, j = l, mid + 1
            while i <= mid or j <= r:
                if i > mid or (j <= r and nums[j] < nums[i]):
                    tmp.append(nums[j])
                    j += 1
                else:
                    tmp.append(nums[i])
                    i += 1
            nums[l: r + 1] = tmp

        mergesort(nums, 0, len(nums) - 1)
        return nums


    def heap_sort(self, nums):
        def build_heap(nums, i, n):
            if i>=n:
                return
            left, right = 2*i+1, 2*i+2
            largest = i
            if left<n and nums[left]>nums[largest]:
                largest = left
            if right<n and nums[right]>nums[largest]:
                largest = right
            if largest!=i:
                nums[i], nums[largest] = nums[largest], nums[i]
                self.build_heap(nums, largest, n)
        
        n = len(nums)
        for i in range(n//2, -1, -1):
            build_heap(nums, i, n)

        for i in range(n-1, -1, -1):
            nums[0], nums[i] = nums[i], nums[0]
            build_heap(nums, 0, i)

        return nums

    def base_sort(self, nums, radix=10):  # 基数排序
        k = int(math.ceil(math.log(max(a), radix)))
        bucket = [ [] for _ in range(radix) ]
        for i in range(1, k+1):
            for val in nums:
                bucket[val%(radix**i)/(radix**(i-1))].append(val)
            del nums[:]
            for b in bucket:
                nums.extend(b)
            bucket = [ [] for _ in range(radix) ]
        return nums

class MaximalSquare:
    # 计算最大矩阵面积
    def maximalSquare(self, matrix):
        m, n = len(matrix), len(matrix[0])
        dp= [ [0]*n for _ in range(m) ]

        for i in range(m):
            if matrix[i][0] == "1":
                dp[i][0] = 1
        for j in range(n):
            if matrix[0][j] == "1":
                dp[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == "0":
                    dp[i][j] = 0
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]) + 1
        return max([max(_) for _ in dp]) ** 2

class Trap:
    # 接雨水问题
    # 查找每个位置(除了第0个和最后一个), 左右两边的最大值，然后取min
    def trap(self, height):
        n = len(height)
        left_dp, right_dp = [0]*n, [0]*n

        for i in range(1, n):
            left_dp[i] = max(height[i-1], left_dp[i-1])

        for i in range(n-2, -1, -1):
            right_dp[i] = max(height[i+1], right_dp[i+1])

        ans = 0
        for i in range(1, n-1):
            if left_dp[i]>height[i] and right_dp[i]>height[i]:
                ans += min(left_dp[i], right_dp[i]) - height[i]
        return ans

class FindNthDigit:
    # 第N位数字
    def findNthDigit(self, n):
        d, count = 1, 9
        while n > d*count:
            n -= d*count
            d += 1
            count *= 10
        index = n-1
        start = 10 ** (d-1)
        num = start + index//d
        digitIndex = index % d
        return num // 10 ** (d-digitIndex-1) % 10
        # return int(str(num)[digitIndex])

class SpiralOrder:
    # 打印螺旋矩阵
    def spiralOrder(self, matrix):
        m, n = len(matrix), len(matrix[0])
        top, buttom, left, right = 0, m-1, 0, n-1
        total = m*n
        ans = []
        while True:
            if len(ans)==total:
                break
            for i in range(left, right+1):
                ans.append(matrix[top][i])
            top += 1
            if top>buttom:
                break

            for i in range(top, buttom+1):
                ans.append(matrix[i][right])
            right -= 1
            if right<left:
                break

            for i in range(right, left-1, -1):
                ans.append(matrix[buttom][i])
            buttom -= 1
            
            if buttom<top:
                break

            for i in range(buttom, top-1, -1):
                ans.append(matrix[i][left])
            left += 1
            if left>right:break
        return ans

class MaxDepth:
    # 二叉树的最大深度
    def maxDepth(self, root):
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right))+1

class IntersectionNode:
    # 相交链表
    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return None
        p, q = headA, headB
        while(p!=q):
            p = p.next if p else headB
            q = q.next if q else headA

        return p

class PruneTree:
    # 二叉树剪枝
    def pruneTree(self, root):
        if not root:
            return None
        root.left = self.pruneTree(root.left)
        root.right= self.pruneTree(root.right)
        if root.left is None and root.right is None and root.val==0:
            return None

        return root

class AddTwoNumbers2:
    def addTwoNumbers(self, l1, l2):
        if not l1 and not l2:
            return 0
        rheadA = self.reverse(l1)
        rheadB = self.reverse(l2)
        cx = 0
        ans = None
        while rheadA or rheadB or cx:
            s = 0
            if rheadA:
                s += rheadA.val
            if rheadB:
                s += rheadB.val
            s += cx
            k = s%10
            cx = s//10
            currentNode = ListNode(k)
            currentNode.next = ans
            ans = currentNode
            if rheadA:
                rheadA = rheadA.next
            if rheadB:
                rheadB = rheadB.next

        return ans

    def reverse(self, head):
        cur, pre = head, None
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

class AddTwoNumbers:
    # 两数和
    def addTwoNumbers(self, l1, l2):
        num1, i = 0, 0
        while l1:
            num1 = l1.val*(10**i) + num1
            l1 = l1.next
        num2, i = 0, 0
        while l2:
            num2 = l2.val*(10**i) + num2
            l2 = l2.next
        s = num1 + num2
        if s==0:
            return ListNode(0)
        head, cur = None, None
        while s:
            cx = s % 10
            if not cur:
                cur = ListNode(cx)
                head = cur
            else:
                cur.next = ListNode(cx)
                cur = cur.next
            s = s//10

        return head

    def addtwonumbers(self, l1, l2):
        dummynode = ListNode()
        cur = dummynode
        cx = 0
        while l1 or l2 or cx:
            _sum = cx
            if l1:
                _sum += l1.val
                l1 = l1.next
            if l2:
                _sum += l2.val
                l2 = l2.next
            node = ListNode(val=_sum%10)
            cx = _sum//10
            cur.next = node
            cur = node
        return dummynode.next

class UniquePaths:
    # 不同路径数
    def uniquePaths(self, m, n):
        dp = [ [0]*n for _ in range(m) ]
        
        for i in range(m):
            dp[i][0] = 1
        for i in range(n):
            dp[0][i] = 1

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[-1][-1]

class UniquePathsWithObstacles:
    # 不同路径 II,(包含障碍物)
    def uniquePathsWithObstacles(self, obstacleGrid):
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [ [0]*n for _ in range(m) ]
        dp[0][0] = 0 if obstacleGrid[0][0]==1 else 1

        for i in range(1, m):
            dp[i][0] = 0 if obstacleGrid[i][0]==1 else dp[i-1][0]

        for i in range(1, n):
            dp[0][i] = 0 if obstacleGrid[0][i]==1 else dp[0][i-1]

        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j]==1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[-1][-1]

class LongestConsecutive:
    # 最长连续序列
    def longestConsecutive(self, nums):
        num_set = set(nums)
        maxlen = 0
        for num in nums:
            if num-1 in nums_set:
                continue
            startnum = num
            tmp = num
            while tmp+1 in num_set:
                tmp = tmp+1
            maxlen = max(maxlen, tmp-startnum+1)
        return maxlen

class IsValid:
    # 有效的括号
    def isValid(self, s):
        dic = {'{': '}',  '[': ']', '(': ')', '?': '?'}
        stack = ['?']
        for c in s:
            if c in dic:
                stack.append(c)
            elif dic[stack.pop()] !=c : return False
        return len(stack)==1

class SubArraySum:
    # 和为K的子数组
    def subarraySum(self, nums, k):
        prefix_sum_cnt = defaultdict(int)
        prefix_sum_cnt[0] = 1   # 特别注意
        cur_sum = 0
        count = 0
        for num in nums:
            cur_sum += num
            count += prefix_sum_cnt[cur_sum-k]
            prefix_sum_cnt[cur_sum] += 1
        return count

class PathSum:
    # 路径总和 III
    def pathsum(self, root, target):
        prefixSum_count = defaultdict(int)
        prefixSum_count[0] = 1  # 特别注意

        def dfs(root, cur_sum):
            if not root:
                return 0
            cnt = 0
            cur_sum += root.val
            cnt += prefixSum_count[cur_sum-target]
            prefixSum_count[cur_sum]+=1
            cnt += dfs(root.left, cur_sum)
            cnt += dfs(root.right, cur_sum)
            prefixSum_count[cur_sum] -= 1   # 退出递归是要-1
            return cnt

        cnt = dfs(root, 0)
        return cnt

class NthUglyNumber:
    def nthUglyNumber(self, n):
        dp = [0] * (n+1)
        dp[0] = 1
        dp[1] = 1
        p2, p3, p5 = 1, 1, 1
        for i in range(2, n):
            dp[i] = min(2*dp[p2], 3*dp[p3], 5*dp[p5])

            if 2*dp[p2]==dp[i]:
                p2+=1
            if 3*dp[p3]==dp[i]:
                p3+=1
            if 5*dp[p5]==dp[i]:
                p5+=1

        return dp[n]

class LongestCommonSubSequence:
    # 最长公共子序列
    def longestCommonSubSequence(self, text1, text2):
        m, n = len(text1), len(text2)
        dp = [ [0]*(n+1) for _ in range(m+1) ]

        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1]==text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[-1][-1]

class MaxPathSum:
    # 二叉树中的最大路径和
    def maxPathSum(self, root):
        self.max = -10000

        def maxroot(root):
            if not root:
                return 0

            left = maxroot(root.left)
            right= maxroot(root.rigth)
            _sum = left + right + root.val
            if _sum > self.max:
                self.max = _sum
            _max == max(left, right)
            if _max+root.val<0:
                return 0
            else:
                return _max + root.val

        maxroot(root)
        return self.max

class MajorityElement:
    # 多数元素 > n/2
    def majorityElement(self, nums):
        cur, cnt = nums[0], 1
        for num in nums[1:]:
            if cnt==0:
                cur = num
                cnt += 1
                continue
            if num==cur:
                cnt += 1
            else:
                cnt -= 1
        return cur
    def majority_element(self, nums):
        nums.sort()
        return nums[len(nums)//2]

class DetectCycle:
    # 环形链表 II
    def detectCycle(self, head):
        if not head:
            return None
        slow, fast = head, head
        while fast is not None:
            slow = slow.next
            if fast.next is None:
                return None
            fast = fast.next.next

            if slow==fast:
                break
        if fast is None:
            return None
        fast = head
        while fast!=slow:
            slow = slow.next
            fast = fast.next

        return fast

class KthLargest:
    def kthLargest(self, root):
        def dfs(root):
            if not root:
                return
            if self.k == 0:
                return
            dfs(root.right)
            self.k -= 1
            if self.k==0:
                self.res = root.val
            dfs(root.left)

        self.k = k
        dfs(root)
        return self.res
            
class CountSmall:
    # 统计右边小数
    def countSmall(self, nums):
        results = []
        if not nums:
            return results

        uniq_list = sorted(list(set(nums)))
        c = [0]*(len(nums)+1)

        for i in range(len(nums)-1, -1, -1):
            _id = self.get_id(uniq_list, nums[i])
            results.append(self.query(c, _id-1))
            self.update(c, _id)

        results.reverse()

        return results

    def lowbit(self, _id):
        return _id & (-_id)

    def get_id(self, uniq_list, num):
        return bisect_left(uniq_list, num)+1

    def query(self, c, _id):
        ret = 0
        while _id>0:
            ret += c[_id]
            _id -= self.lowbit(_id)

    def update(self, c, _id):
        while _id < len(c):
            c[_id] += 1
            _id += self.lowbit(_id)

class MaxProfit:
    # 买卖股票最佳时间
    def maxProfit(self, prices):
        ans = 0
        for i in range(1, len(prices)):
            ans += max(prices[i]-prices[i-1], 0)
        return ans

class FindLength:
    # 查找最长重复子数组
    def findLength(self, nums1, nums2):
        m, n = len(nums1), len(nums2)
        dp = [ [0]*(n+1) for _ in range(m+1) ]

        ans = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    ans = max(ans, dp[i][j])

        return ans

class LongestValidParatheses:
    # 最长有效括号
    def longestValidParentheses(self, s):
        stack = [-1]
        _max = 0
        for i,c in enumerate(s):
            if len(stack)==1:
                stack.append(i)
                continue
            if c=="(":
                stack.append(i)
            else:
                if s[stack[-1]]=='(':
                    stack.pop(-1)
                    _max = max(_max, i-stack[-1])
                else:
                    stack.append(i)

        return _max

class FindKthNumber:
    # 字典序的第k小数字
    def findKthNumber(self, n, k):
        def getsteps(cur, n):
            steps, first, last = 0, cur, cur
            while first<=n:
                steps += min(n, last)-first + 1
                first *= 10
                last = last*10 + 9

            return steps

        
        cur = 1
        k = k-1
        while k:
            steps = getsteps(cur, n)
            if steps<=k:
                k-=steps
                cur += 1
            else:
                cur *= 10
                k -= 1
        return cur

class KthSmallest:
    # 矩阵第k小的数
    def kthSmallest(self, matrix, k):
        n = len(matrix)

        def smallHalf(mid):
            i, j = n-1, 0
            count = 0
            while i>=0 and j<n:
                if matrix[i][j] <= mid:
                    count += i+1
                    j += 1
                else:
                    i -= 1
            return count

        lo, hi = matrix[0][0], matrix[-1][-1]
        while lo<hi:
            mid = (lo+hi) // 2
            if smallHalf(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return hi

class PreorderTraversal:
    # 前序遍历
    def preorderTraver(self, root):
        self.ans = []
        if not root:
            return self.ans

        def dfs(self, root):
            if not root:
                return
            self.ans.append(root.val)
            dfs(root.left)
            dfs(root.right)

        dfs(root)
        return self.ans

    def preorderTraver(self, root):
        self.ans = []
        if not root:
            return []
        stack = []
        stack.append(root)
        node = root
        while stack or node:
            while node:
                self.ans.append(node.val)
                stack.append(node)
                node = node.left
            node = stack.pop()
            node = node.right
        return self.ans

class InorderTraver:
    def inorderTraver(self, root):
        self.ans = []
        if not root:
            return []
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            self.ans.append(root.val)
            dfs(root.right)
        dfs(root)
        return self.ans

    def inorderTraver(self, root):
        self.ans = []
        if not root:
            return []
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            self.ans.append(root.val)
            root = root.right
        return self.ans

class FindMaxLength:
    # 连续数组
    def findmaxLength(self, nums):
        # 使用前缀和处理
        prefix_counter = dict()
        n = len(nums)
        counter = 0
        prefix_counter[counter] = -1
        maxlen = 0
        for i in range(n):
            if nums[i]==1:
                counter += 1
            else:
                counter -= 1
            if counter not in prefix_counter:
                prefix_counter[counter] = i
            else:
                pi = prefix_counter[counter]
                maxlen = max(maxlen, i-pi)
        return maxlen

class BuildTree:
    # 由前序中序遍历构造树
    def buildTree(self, preorder, inorder):
        n = len(inorder)
        index = {ele:i for i, ele in enumerate(inorder)}

        def build(lpre, rpre, lin, rin):
            if lpre>rpre:
                return None
            preorder_root = lpre
            inorder_root = index[preorder[preorder_root]]
            root = TreeNode(preorder[preorder_root])

            size_left_subtree = inorder_root - lin
            root.left = build(lpre+1, lpre+size_left_subtree, lin, inorder_root-1)
            root.right= build(lpre+size_left_subtree+1, rpre, inorder_root+1, rin)

            return root

        return build(0, n-1, 0, n-1)

    # 由中序和后序遍历构造树
    def buildTree(self, inorder, postorder):
        def build(in_left, in_right):
            if in_left>in_right:
                return None

            val = postorder.pop()
            root= TreeNode(val)
            
            index = inx_map(val)

            # 右子树, 要先构建右子树(因为postorder pop顺序)
            root.right = build(index+1, in_right)
            # 左子树
            root.left = build(in_left, index-1)
            return root

        inx_map = {val:idx for idx,val in enumerate(inorder)}
        return build(0, len(inorder)-1)

class ReverseBetween:
    # 中间反转链表
    def reverseBetween(self, head, left, right):
        def reverse_link(self, head):
            pre, cur = None, head
            while cur:
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt

        dummynode = ListNode(-1)
        dummynode.next= head
        pre = dummynode
        for i in range(left-1):
            pre = pre.next

        right_node = pre
        for i in range(right-left+1):
            right_node = right_node.next
        left_node = pre.next
        cur = right_node.next
        pre.next = None
        right_node.next = None
        reverse_link(left_node)
        pre.next = right_node
        left_node.next = cur

        return dummynode

class TriangleNumber:
    # 三角形个数
    def triangleNumber(self, nums):
        nums = sorted(nums)
        n = len(nums)
        ans = 0
        for i in range(n):
            k = i
            for j in range(i+1, n):
                while k+1<n and nums[k+1]<(nums[i]+nums[j]):
                    k += 1
                ans += max(k-j, 0)

        return ans

class TwoSum:
    def twoSum(self, nums, target):
        n = len(nums)
        i, j = 0, n-1
        while i<j:
            t = nums[i]+nums[j]
            if t == target:
                return  [i+1, j+1]
            elif t < target:
                i += 1
            else:
                j -= 1
        return [-1, -1]

class FindMinHeightTrees:
    # 查找最小高度树
    # 从叶子节点找根
    def findMinHeightTrees(self, n, edges):
        if n<=1:
            return [0]

        node_degree = defaultdict(int)
        node_edges = defaultdict(list)
        for s, e in edges:
            node_degree[s] += 1
            node_degree[e] += 1
            node_edges[s].append(e)
            node_edges[e].append(s)

        stack = [i for i in range(n) if node_degree[i]==1]
        res = []
        while stack:
            res = []
            size = len(stack)
            for i in range(size):
                k= stack.pop(0)
                res.append(k)
                for e in node_edges[k]:
                    node_degree[e]-=1
                    if node_degree[e]==1:
                        stack.append(e)
        return res

class FindTarget:
    # 从二叉搜索树中查找和为target的两个数
    def findTarget(self, root, target):
        self.nums = []
        self.inorder(root)
        left, right = 0, len(self.nums)-1
        while left<right:
            if self.nums[left]+self.nums[right]==target:
                return True
            elif self.nums[left]+self.nums[right]<target:
                left += 1
            else:
                right -= 1
        return False

    def inorder(self, root):
        while not root:
            return 
        self.inorder(root.left)
        self.nums.append(root.val)
        self.inorder(root.right)

class HasCycle:
    # 判断是否有环
    def hasCycle(self, head):
        if not head or not head.next:
            return False
        slow, fast = head, head.next
        while slow!=fast:
            if not fast.next or not fast.next.next:
                return False
            slow = slow.next
            fast = fast.next.next
        return True

class MaximumSwap:
    def maximumSwap(self, num):
        digits = []
        while num:
            digits.append(num % 10)
            num = num//10
        digits.reverse()
        for i in range(len(digits)):
            max_d, k = digits[i], i
            for j in range(len(digits)-1, i, -1):
                if digits[j]>max_d:
                    max_d, k = digits[j], j
            if k!=i:
                digits[i], digits[k] = digits[k], digits[i]
                break

        ans = 0
        for n in digits:
            ans = ans*10 + n
        return ans

class PathSum:
    def pathSum(self, root, target):
        if not root:
            return []
        self.ans = []
        def dfs(root, _sum, path):
            if not root:
                return
            _sum += root.val
            path.append(root.val)
            if not root.left and not root.right:
                if _sum==target:
                    self.ans.append(path[:])
            dfs(root.left, _sum, path)
            dfs(root.right, _sum, path)
            _sum -= root.val
            path.pop()
        _sum = 0
        path = []
        dfs(root, _sum, path)
        return self.ans

class FindNumberOfLIS:
    # 最长递增子序列个数
    def findNumberOfLIS(self, nums):
        n = len(nums)
        dp = [1] * n
        cnt = [1]*n
        ans, maxlen = 0, 0
        for i in range(n):
            for j in range(i):
                if nums[i]>nums[j]:
                    if dp[i] < dp[j]+1:
                        dp[i] = dp[j]+1
                        cnt[i]= cnt[j]
                    elif dp[i]==dp[j]+1:
                        cnt[i] += dp[j]
                if dp[i]>maxlen:
                    maxlen = dp[i]
                    ans = cnt[i]
                elif dp[i]==maxlen:
                    ans += cnt[i]
        return ans

class RemoveNthFromEnd:
    # 删除链表的倒数第N个节点
    def removeNthFromEnd(self, head, n):
        dummy = ListNode(-1)
        dummy.next = head
        slow, fast = dummy, head
        for i in range(n):
            fast = fast.next

        while fast:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next

        return dummy.next

class PostOrderTraversal:
    def postorderTraversal(self, root):
        if not root:
            return []

        ans = []
        stack = []
        prev = None
        while root or stack:
            while root:
                stack.appen(root)
                root = root.left
            root = stack.pop()
            if not root.right or root.right==prev:
                ans.append(root.val)
                prev = root
                root = None
            else:
                stack.append(root)
                root = root.right
        return ans

class Rand10:
    # 由rand7() -> rand10()
    def rand10(self):
        while True:
            a, b = rand7(), rand7()
            x = (a-1)*7 + b
            if x<=40:
                return 1+x%10

class FindMin:
    # 寻找旋转排序数组中的最小值
    def findMin(self, nums):
        left, right = 0, len(nums)-1
        while left<right:
            mid = left+(right-left)//2
            if nums[mid]<nums[right]:
                right = mid
            else:
                left = mid+1
        return nums[left]

class LongestIncreasingPath:
    # 矩阵中的最长递增路径
    def __init__(self):
        self.dirs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    def longestIncreasingPath(self, matrix):
        @lru_cache()
        def dfs(row, col):
            largest_length = 1
            for dx, dy in self.dirs:
                _row, _col = row+dx, col+dy
                if (_row>=0 and _row<m and _col>=0 and _col<n) and \
                   matrix[_row][_col]>matrix[row][col]:
                    largest_length = max(largest_length, dfs(_row, _col)+1)

            return largest_length

        m, n = len(matrix), len(matrix[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i,j))
        return ans

    def longestIncreasingPath(self, matrix):
        @lru_cache
        def dfs(i, j):
            largest_length = 1
            for (dx, dy) in self.dirs:
                _i, _j = i+dx, j+dy
                if (_i>=0 and _i<m and _j>=0 and _j<n) and \
                   matrix[_i][_j] > matrix[i][j]:
                    largest_length = max(largest_lenght, dfs(_i, _j)+1)
            return largest_length
        m, n = len(matrix), len(matrix[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i, j))
        return ans

class MinEatingSpeed:
    # 爱吃香蕉的小猴
    def minEatingSpeed(self, piles, h):
        def get_cost(k):
            t = 0
            for p in piles:
                t += (p//k) if p%k==0 else (p//k)+1
            return t

        if h==len(piles):
            return max(piles)
        l, r = 1, max(piles)
        while l<r:
            mid = (l+r) // 2
            if get_cost(mid)<=h:
                r = mid
            else:
                l = mid+1
        return l

class SearchMatirx:
    # 矩阵搜索
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        i, j = 0, len(matrix[0])-1
        while i<=len(matrix)-1 and j>=0:
            if matrix[i][j]==target:
                return True
            if matrix[i][j] < target:
                i += 1
            else:
                j -= 1
        return False

class NextPermutation:
    # 下一个排列
    def nextPermutation(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i, j = n-2, n-1
        while i>=0:
            if nums[i]<nums[j]:  # 找出第一个前一个数小于后一个数的位置
                break
            i-=1
            j-=1

        if i<0:                 # 表明该数组序列已经排好序(逆向)
            nums.reverse() 
            return nums
        
        k = n-1
        while (k>=j):
            # 找出后面第一个大于i的位置, 并交换
            if nums[k]<=nums[i]:
                k -= 1
            else:
                nums[k], nums[i] = nums[i], nums[k]
                break

        l, r = j, n-1
        while l<r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
        return nums

class CanPartitionKSubsets:
    # 划分为k个相等的子集, <使用回溯>
    def canPartitionKSubsets(self, nums, k):
        def dfs(i):
            if i == len(nums):
                return True
            for j in range(k):
                if j and cur[j] == cur[j-1]:
                    continue
                cur[j] += nums[i]
                if cur[j] <= s and dfs(i + 1):
                    return True
                cur[j] -= nums[i]
            return False

        s, mod = divmod(sum(nums), k)
        if mod:
            return False
        cur = [0] * k
        nums.sort(reverse=True)
        return dfs(0)

class FindClosestElements:
    # 查找K个最接近的元素
    def findClosestElements(self, nums, k, x):
        # 查找出离x最近的元素
        n = len(nums)
        if n<k:
            return nums

        inx = bisect_left(nums, x)
        left,right = inx-1, inx
        for i in range(k):
            if left<0:
                right += 1
            elif right>n or x-nums[left]<=nums[right]-x:
                left -= 1
            else:
                right += 1
        return nums[left:right]

class BinarySearch:
    # 二分查找
    def search(self, nums, target):
        n = len(nums)
        left, right = 0, n-1
        while left<=right:
            mid = left + (right-left)//2
            if nums[mid]==target:
                return mid
            if nums[mid]<=target:
                left = mid+1
            else:
                right = mid-1

        return -1

class LastRemaining:
    # 环形最后保留数
    def lastRemaining(self, n, m):
        pos = 0
        for i in range(2, n + 1):
            pos = (pos + m) % i
        return pos

class Inorder:
    # 中序遍历
    def inorder(self, root):
        if not root:
            return
        stack = []
        ans = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            ans.append(root.val)
            root = root.right
        return ans


class Node:
    def __init__(self, key="", count=0):
        self.prev = None
        self.next = None
        self.keys = {key}
        self.count = count

    def insert(self, node: 'Node') -> 'Node':  # 在 self 后插入 node
        node.prev = self
        node.next = self.next
        node.prev.next = node
        node.next.prev = node
        return node

    def remove(self):  # 从链表中移除 self
        self.prev.next = self.next
        self.next.prev = self.prev
class AllOne:
    # 全O(1)数据结构
    def __init__(self):
        self.root = Node()
        self.root.prev = self.root
        self.root.next = self.root  # 初始化链表哨兵，下面判断节点的 next 若为 self.root，则表示 next 为空（prev 同理）
        self.nodes = {}

    def inc(self, key: str) -> None:
        if key not in self.nodes:  # key 不在链表中
            if self.root.next is self.root or self.root.next.count > 1:
                self.nodes[key] = self.root.insert(Node(key, 1))
            else:
                self.root.next.keys.add(key)
                self.nodes[key] = self.root.next
        else:
            cur = self.nodes[key]
            nxt = cur.next
            if nxt is self.root or nxt.count > cur.count + 1:
                self.nodes[key] = cur.insert(Node(key, cur.count + 1))
            else:
                nxt.keys.add(key)
                self.nodes[key] = nxt
            cur.keys.remove(key)
            if len(cur.keys) == 0:
                cur.remove()

    def dec(self, key: str) -> None:
        cur = self.nodes[key]
        if cur.count == 1:  # key 仅出现一次，将其移出 nodes
            del self.nodes[key]
        else:
            pre = cur.prev
            if pre is self.root or pre.count < cur.count - 1:
                self.nodes[key] = cur.prev.insert(Node(key, cur.count - 1))
            else:
                pre.keys.add(key)
                self.nodes[key] = pre
        cur.keys.remove(key)
        if len(cur.keys) == 0:
            cur.remove()

    def getMaxKey(self) -> str:
        return next(iter(self.root.prev.keys)) if self.root.prev is not self.root else ""

    def getMinKey(self) -> str:
        return next(iter(self.root.next.keys)) if self.root.next is not self.root else ""

class LargestRectangleArea:
    # 柱状图中的最大矩形面积: 左右比当前小的位置，area = height_i * (right-left)
    def largestRectangleArea(self, heights):
        # 使用单调栈
        n = len(heights)
        stack = []
        right_stack = []
        for i in range(n-1, -1, -1):
            while stack and heights[stack[-1]]>=heights[i]:
                stack.pop()
            if stack:
                right_stack.append(stack[-1])
            else:
                right_stack.append(n)
            stack.append(i)
        right_stack.reverse()
        
        stack = []
        left_stack = []
        for i in range(n):
            while stack and heights[stack[-1]]>=heights[i]:
                stack.pop()
            if stack:
                left_stack.append(stack[-1])
            else:
                left_stack.append(-1)
            stack.append(i)

        max_area = 0
        for i in range(n):
            max_area = max(max_area, heights[i]*(right_stack[i]-left_stack[i]-1))
        return max_area

class RightSideView:
    # 二叉树的右视图
    def rightSideView(self, root):
        # 使用层次遍历，取最右
        if not root:
            return [] 
        stack = [root]
        ans = []
        while stack:
            size = len(stack)
            tmp = []
            for i in range(size):
                root = stack.pop(0)
                tmp.append(root.val)
                if root.left:
                    stack.append(root.left)
                if root.right:
                    stack.append(root.right)
            ans.append(tmp[-1])

        return ans
    def rightSideView(self, root):
        if not root:
            return []
        stack = [root]
        ans = []
        while stack:
            size = len(stack)
            for i in range(size):
                node = stack.pop(0)
                if i==(size-1):
                    ans.append(node.val)
                if node.left: stack.append(node.left)
                if node.right:stack.append(node.right)
        return ans

class MinRefuelStops:
    # 最少加油次数
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        # 贪心
        n = len(stations)
        ans, fuel, prev, h = 0, startFuel, 0, []
        for i in range(n + 1):
            curr = stations[i][0] if i < n else target
            fuel -= curr - prev
            while fuel < 0 and h:
                fuel -= heappop(h)
                ans += 1
            if fuel < 0:
                return -1
            if i < n:
                # 使用优先队列存储
                heappush(h, -stations[i][1])
                prev = curr
        return ans
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        # 使用动态规划
        n = len(stations)
        dp = [0]*(n+1)   # dp[i] 表示加i次油最大的距离
        dp[0] = startFuel
        for i in range(n):
            for j in range(i, -1, -1):
                if dp[j] >= stations[i][0]:
                    dp[j+1] = max(dp[j+1], dp[j]+stations[i][1])
        for i in range(n+1):
            if dp[i] >= target:
                return i
        return -1

class SmallestRange:
    def smallestRange(self, nums):
        range_left, range_right = -10**9, 10**9
        max_value = max(vec[0] for vec in nums)
        priority_queue = [(vec[0], i, 0) for i,vec in enumerate(nums)]
        heapq.heapify(priority_queue)

        while True:
            min_value, row, inx = heapq.heappop(priority_queue)
            if max_value-min_value < range_right-range_left:
                range_left, range_right = min_value, max_value
            if idx == len(row)-1:
                break
            max_value = max(max_value, nums[row][idx+1])
            heapq.heappush(priority_queue, (nums[row][idx+1], row, idx+1))

        return [range_left, range_right]

class MinInteger:
    # 最多交换k次使得数字最小
    class BIT:
        def __init__(self, n):
            self.n = n
            self.tree = [0]*(n+1)

        def lowbit(self, x):
            return x & (-x)

        def query(self, _id):
            ans = 0
            while _id > 0:
                ans += self.tree[_id]
                _id -= self.lowbit(_id)
            return ans

        def update(self, _id):
            while _id<=self.n:
                self.tree[_id] += 1
                _id += self.lowbit(_id)

    def minInteger(self, num, k):
        # 统计每个
        n = len(num)
        bit = BIT(n)
        pos = [list() for _ in range(10)]
        for i in range(n-1, -1, -1):
            pos[ord(num[i])-ord('0')].append(i)

        ans = ''
        for i in range(1, n+1):
            for j in range(10):
                if pos[j]:
                    # 计算交互次数
                    behind_change = bit.query(pos[j][-1]+1) - bit.query(n)
                    chang_count = pos[j][-1] + behind_change - i
                    if chang_count <= k:
                        bit.update(pos[j][-1])
                        pos[j].pop()
                        ans += str(j)
                        k -= chang_count
                        break

        return ans

class FirstMissingPositive:
    # 第一个缺失的正数
    def firstMissingPositive(self, nums):
        n = len(nums)
        
        # 1. 将负数变为正数(n+1)：3,4,-1,1,9,-5 ==> 3,4,7,1,9,7
        for i in range(n):
            if nums[i]<=0:
                nums[i] = n+1

        # 2. 将<=6的元素对应位置变为负数：3,4,-1,1,9,-5 ==> -3,4,-7,-1,9,7
        for i in range(n):
            num = abs(nums[i])
            if num<=n:
                nums[num-1] = -abs(nums[num-1])

        # 3. 返回第一个大于0的元素下标+1
        for i in range(n):
            if nums[i]>0:
                return i+1
        return n+1


class Node:
    def __init__(self, key=0, value=0):
        self.key = key
        self.val = val
        self.pre = None
        self.next = None
class LRUCache:
    # 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
    def __init__(self, capacity):
        self.capacity = capacity
        self.container= dict()
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.pre  = self.head
        self.size = 0

    def get(self, key):
        node = self.container.get(key, None)
        if node is None:return -1
        self.move2head(node)
        return node.val

    def put(self, key, value):
        if key not in self.container:
            node = Node(key=key, value=value)
            self.add2head(node)
            self.size += 1
            while self.size>capacity:
                self.delete_node()
                self.size -= 1
        else:
            node = self.container[key]
            node.val = value
            self.move2head(node)

    def move2head(self, node):
        node.pre.next = node.next
        node.next.pre = node.pre
        self.add2head(node)
    def add2head(self, node):
        node.pre = self.head
        node.next = self.head.next
        self.head.next.pre = node
        self.head.next = node
    def delete_node(self):
        node = self.tail.pre
        node.pre.next = self.tail
        self.tail.pre = node.pre
        self.container.pop(node.key)

class IsValidBST:
    # 验证二叉搜索树
    def isValidBST(self, root):
        return self.isvalid(root)

    def isvalid(self, root, low=float('-inf'), upper=float('inf')):
        if not root:
            return True
        if root.val <= low or root.val>=upper:
            return False

        return self.isvalid(root.left, low, root.val) and \
                self.isvalid(root.right, root.val, upper)

class MaxSubArray:
    # 最大子数组和
    def maxSubArray(self, nums):
        n = len(nums)
        dp = [0]*n
        dp[0] = nums[0]

        for i in range(1, n):
            if dp[i-1]>=0:
                dp[i] = dp[i-1]+nums[i]    # 一个数加上>=0的数一定大于等于自己本身
            else:
                dp[i] = nums[i]

        return max(dp)

    def maxsubarray(self, nums):
        n = len(nums)
        _max, pre = nums[0], nums[0]

        for i in range(1, n):
            pre = max(pre+nums[i], nums[i])
            _max= max(pre, _max)
        return _max
        
class NumIslands:
    # 岛屿数量
    def numIslands(self, grid):
        self.cnt = 0
        m, n = len(grid), len(grid[0])
        def dfs(grid, i, j):
            if i<0 or i>=m or j<0 or j>=n:
                return
            if grid[i][j]!="1":
                return
            grid[i][j] = "2"
            dfs(grid, i-1, j)
            dfs(grid, i, j-1)
            dfs(grid, i+1, j)
            dfs(grid, i, j+1)

        for i in range(m):
            for j in range(n):
                if grid[i][j]=="1":
                    self.cnt += 1
                    dfs(grid, i,j)
        return self.cnt

class SortOddEvenList:
    # 排序奇升偶降链表
    class ListNode:
        def __init__(self, val):
            self.val = val
            self.next = next

    def sortOddEvenList(self, head):
        if not head or not head.next:
            return head

        odd_list, even_list = self.partion(head)
        even_list = self.reverse(even_list)
        return self.merge(odd_list, even_list)

    def partion(self, head):
        evenHead = head.next
        odd, even = head, evenHead
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = None
        return head,evenHead

    def reverse(self, head):
        pre, cur = head, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

    def merge(self, p, q):
        dummy_node = ListNode(-1)
        head = dummy_node
        while p and q:
            if p.val<q.val:
                head.next = p
                p = p.next
            else:
                head.next = q
                q = q.next
            head = head.next
        if p:
            head.next = p
        if q:
            head.next = q
        return dummy_node.next

class KthSmallest:
    # 二叉搜索树第k小的元素值
    def kthSmallest(self, root, k):
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k==0:
                return root.val
            root = root.right

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while l<= r:
            mid = (l+r) // 2
            # [l, mid-1] mid [mid+1, r]
            if nums[mid] < target:
                l = mid +1
            if nums[mid] > target:
                r = mid -1
            if nums[mid] == target:
                r = mid -1
        left = l

        l, r = 0, len(nums)-1
        while l<= r:
            mid = (l+r) // 2
            # [l, mid-1] mid [mid+1, r]
            if nums[mid] < target:
                l = mid +1
            if nums[mid] > target:
                r = mid -1
            if nums[mid] == target:
                l = mid +1
        right = r
        return right - left + 1

class detectCycle:
    def detectCycle(self, head):
        slow, fast = head, head
        while slow:
            if not(fast and fast.next): return
            slow, fast = slow.next, fast.next.next
            if slow==fast:
                break
        fast=head
        while fast!=slow:
            fast, slow = fast.next, slow.next
        return fast

class MaximalSquare:
    # 计算最大正方形面积
    def maximalSquare(self, matrix):
        m, n = len(matrix), len(matrix[0])
        dp = [ [0]*n for _ in range(m) ]
        for i in range(m):
            if matrix[i][0] == "1":
                dp[i][0] = 1
        for i in range(n):
            if matrix[0][i] == "1":
                dp[0][i] = 1

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "0":
                    dp[i][j] = 0
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]) + 1
        return max([max(_) for _ in dp])**2


class Multiply:
    # 两个字符串数字相乘
    def multiply(self, num1, num2):
        if num1=="0" or num2=="0":
            return "0"

        ans = "0"
        m, n = len(num1), len(num2)
        for i in range(n-1, -1, -1):
            add = 0
            y = int(num2[i])
            curr = ["0"] * (n-1-i)
            for j in range(m-1, -1, -1):
                product=int(num1[j]) * y + add
                curr.append(str(product % 10))
                add = product // 10
            if add>0:
                curr.append(str(add))
            curr = "".join(cur[::-1])
            ans = self.add_string(ans, curr)

        return ans

    def add_string(self, num1, num2):
        i, j = len(num1)-1, len(num2)-1
        add = 0
        ans = list()
        while i>=0 or j>=0 or add:
            x = int(num1[i]) if i>=0 else 0
            y = int(num2[j]) if j>=0 else 0
            result = x+y+add
            ans.append(str(result % 10))
            add = result // 10
            i -= 1
            j -= 1
        return "".join(ans[::-1])

class ThreeSumClosest:
    # 最接近三数之和
    def threeSumClosest(self, nums, target):
        nums = sorted(nums)
        n = len(nums)
        nearest = -1
        for i, num in enumerate(nums):
            if i>0 and nums[i]==nums[i-1]: continue
            left, right = i+1, n-1
            while left<right:
                s = nums[i] + nums[left] + nums[right]
                if s == target:
                    return target
                if abs(s-target) < abs(nearest-target):
                    nearest = s
                if s>target :
                    right = right-1
                    while left<right and nums[right]==nums[right+1]:
                        right -= 1
                else:
                    left = left+1
                    while left<right and nums[left]==nums[left-1]:
                        left += 1
        return nearest

class LeverOrder:
    # 二叉树层序遍历 left->right
    def leverOrder(self, root):
        if not root:
            return []
        stack = [root]
        ans = []
        while stack:
            size = len(stack)
            tmp = []
            for i in range(size):
                root = stack.pop(0)
                tmp.append(root.val)
                if root.left:
                    stack.append(root.left)
                if root.right:
                    stack.append(root.right)
            ans.append(tmp)
        return ans

class FindKthLargest:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 使用快排思想
        def partion(nums, left, right):
            pivot = random.randint(left, right)
            nums[pivot], nums[right] = nums[right], nums[pivot]
            i = left-1
            for j in range(left, right):
                if nums[j]<nums[right]:
                    i += 1
                    nums[i], nums[j] = nums[j], nums[i]

            i += 1
            nums[i], nums[right] = nums[right], nums[i]

            return i

        def helper(nums, index, left, right):
            i = partion(nums, left, right)
            if i == index:
                return nums[i]
            if i>index:
                return helper(nums, index, left, i-1)
            else:
                return helper(nums, index, i+1, right)

        return helper(nums, len(nums)-k, 0, len(nums)-1)

class HasPathSum:
    # 路径总和
    def hasPathSum(self, root, targetSum):
        if not root:
            return False
        if not root.left and not root.right:
            return targetSum==root.val
        return self.hasPathSum(root.left, targetSum-root.val) or \
                self.hasPathSum(root.right, targetSum-root.val)

    def hasPathSum(self, root, targetSum):
        if not root:
            return False
        root_dq = collections.deque([root])
        val_dq  = collections.deque([root.val])
        while root_dq:
            node = root_dq.popleft()
            val = val_dq.popleft()
            if not node.left and not node.right:
                if val == targetSum:
                    return True
                continue
            if node.left:
                root_dq.append(node.left)
                val_dq.append(val+node.left.val)
            if node.right:
                root_dq.append(node.right)
                val_dq.append(val+node.right.val)

class MedianFinder:
    # 流方式计算中位数
    def __init__(self):
        self.queMin = list()
        self.queMax = list()

    def addNum(self, num: int) -> None:
        queMin_ = self.queMin
        queMax_ = self.queMax

        if not queMin_ or num <= -queMin_[0]:
            heapq.heappush(queMin_, -num)
            if len(queMax_) + 1 < len(queMin_):
                heapq.heappush(queMax_, -heapq.heappop(queMin_))
        else:
            heapq.heappush(queMax_, num)
            if len(queMax_) > len(queMin_):
                heapq.heappush(queMin_, -heapq.heappop(queMax_))
        
    def findMedian(self) -> float:
        queMin_ = self.queMin
        queMax_ = self.queMax

        if len(queMin_) > len(queMax_):
            return -queMin_[0]
        return (-queMin_[0] + queMax_[0]) / 2

import queue
class MaxQueue:
    # 请定义一个队列并实现函数 max_value 得到队列里的最大值，
    # 要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)
    def __init__(self):
        self.deque = queue.deque()
        self.queue = queue.Queue()

    def max_value(self) -> int:
        return self.deque[0] if self.deque else -1

    def push_back(self, value: int) -> None:
        while self.deque and self.deque[-1]<value:
            self.deque.pop()
        self.deque.append(value)
        self.queue.put(value)

    def pop_front(self) -> int:
        if self.queue.qsize()<=0:
            return -1
        ans = self.queue.get()
        if ans==self.deque[0]:
            self.deque.popleft()
        return ans

class Codec:
    def serialize(self, root):
        if not root:
            return ''
        return str(root.val) + ',' + self.serialize(root.left) + \
                ',' + self.serialize(root.right)
    def serializev2(self, root):
        ans = []
        if not root:
            return '[]'
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node:
                ans.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                ans.append('None')
        return f"[{','.join(ans)}]"

    def deserialize(self, data):
        def dfs(datalist):
            value = datalist.pop(0)
            if not value:
                return
            root = TreeNode(int(value))
            root.left = dfs(datalist)
            root.right= dfs(datalist)
            return root

        datalist = data.split(',')
        if not datalist:
            return None
        return dfs(datalist)

    def deserializev2(self, data):
        datalist = data[1:-1].split(',')
        if len(datalist)<1 or not datalist[0]:
            return None
        root = TreeNode(int(datalist[0]))
        queue = deque([root])
        i = 1
        while queue:
            node = queue.popleft()
            if datalist[i] != "None":
                node.left = TreeNode(int(datalist[i]))
                queue.append(node.left)
            i += 1
            if datalist[i] != "None":
                node.right = TreeNode(int(datalist[i]))
                queue.append(node.right)
            i += 1
        return root

class ReverseBetween:
    def reverseBetween(self, head, left, right):
        # 头插法
        dummy_node = ListNode(-1)
        dummy_node.next = head
        pre = dummy_node
        for i in range(left-1):
            pre = pre.next

        cur = pre.next
        for i in range(right-left):    # 整个过程中cur是不变的
            nxt = cur.next
            cur.next = nxt.next
            nxt.next = pre.next
            pre.next = nxt

        return dummy_node.next

class invertTree:
    def invertTree(self, root):
        if not root:
            return
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root

class BiSearch:
    def search(self, nums, target):
        left, right = 0, len(nums)-1
        while left<=right:
            mid = (left+rigth) // 2
            if nums[mid]==target:
                return True
            
            if nums[mid]>target:
                right = mid-1
            else:
                left = mid+1
        return False

class GetPermutation:
    # 获取所有排列组合中的第k个
    def getPermutation(self, n, k):
        self.ans = []

        def dfs(nums, visited, path, depth, size):
            if depth==size:
                self.ans.append(path[:])
                return 

            for i in range(size):
                if visited[i]:
                    continue
                path.append(nums[i])
                visited[i] = True
                dfs(nums, visited, path, depth+1, size)
                if len(self.ans)==k:
                    return
                visited[i] = False
                path.pop()

        path = []
        nums = [str(i) for i in range(1, n+1)]
        visited = [False]*len(nums)
        for i in range(len(nums)):
            dfs(nums, visited, path, i, len(nums))
        return ''.join(self.ans[k-1])

class GetPermutation:
    # 第k个排列, 使用剪枝处理
    def getPermutation(self, n, k):
        def dfs(n, k, index, path):
            if index==n:
                return
            cnt = factorial[n-1-index]
            for i in range(1, n+1):
                if visited[i]: continue
                if cnt < k:
                    k -= cnt
                    continue
                path.append(i)
                visited[i] = True
                dfs(n, k, index+1, path)
                return
        if n==0:
            return ""

        path = []
        visited = [False] * n
        factorial = [1] * n
        for i in range(2, n+1):
            factorial[i] = factorial[i-1] * i

        dfs(n, k, 0, path)
        return ''.join([str(num) for num in path])

class LongestPalindromeSubseq:
    # 最长可能回文串
    def longestPalindromeSubseq(self, s):
        n = len(s)
        dp = [[0]*n for _ in range(n)]
        for i in range(n-1, -1, -1):
            dp[i][i]=1
            for j in range(i+1, n):
                if s[i]==s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])

        return dp[0][n-1]

class MinStack:
    # 最小栈
    def __init__(self):
        self.stack = []
        self.min_stack = [100000000000]

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]

class Search:
    # 搜索旋转排序数组
    def search(self, nums, target):
        left, right = 0, len(nums)-1
        while left<=right:
            mid = left+(right-left)//2
            if nums[mid] == target:
                return mid
            if nums[0]<=nums[mid]:
                if nums[0]<=target<nums[mid]:
                    right = mid-1
                else:
                    left = mid+1
            else:  # nums[0]>nums[mid] 表明存在逆序
                if nums[mid]<target<=nums[-1]:
                    left = mid+1
                else:
                    right = mid-1
        return -1

class PathSum:
    def pathSum(self, root, targetSum):
        self.ans = []

        def dfs(root, path, _sum):
            if not root:
                return 

            _sum += root.val
            path.append(root.val)
            if not root.left and not root.right:
                if _sum==targetSum:
                    self.ans.append(path[:])

            dfs(root.left, path, _sum)
            dfs(root.right, path, _sum)
            _sum -= root.val
            path.pop()

        path = []
        _sum = 0
        dfs(root, path, _sum)
        return self.ans

class PathSumIII:
    # 路径和
    def pathSum(self, root, targetSum):
        prefixSum_count = defaultdict(int)
        prefixSum_count[0]=1

        def dfs(root, cur_sum):
            if not root:
                return 0
            cnt = 0
            cur_sum += root.val
            cnt += prefixSum_count[cur_sum-targetSum]
            prefixSum_count[cur_sum]+=1
            cnt += dfs(root.left, cur_sum)
            cnt += dfs(root.right, cur_sum)
            prefixSum_count[cur_sum] -= 1
            return cnt

        cnt = dfs(root, 0)
        return cnt

class ReorderList:
    def reorderList(self, head):
        if not head:
            return 
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        l1, l2 = head, slow.next
        slow.next = None
        rl2 = self.reverse(l2)
        self.merge(l1, rl2)

    def reverse(self, head):
        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

    def merge(self, l1, l2):
        while l1 and l2:
            l1_next = l1.next
            l2_next = l2.next

            l1.next = l2
            l1 = l1_next

            l2.next = l1
            l2 = l2_next

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def sortFunc(head: ListNode, tail: ListNode) -> ListNode:
            if not head:
                return head
            if head.next == tail:
                head.next = None
                return head
            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            return merge(sortFunc(head, mid), sortFunc(mid, tail))

        def merge(head1: ListNode, head2: ListNode) -> ListNode:
            dummyHead = ListNode(0)
            temp, temp1, temp2 = dummyHead, head1, head2
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return dummyHead.next

        return sortFunc(head, None)

class SortList:
    def sortList(self, head):
        if not head or not head.next:
            return head
        dummynode = ListNode(-1)
        dummynode.next = head

        return self.quick_sort(dummynode, None)

    def quick_sort(self, head, end):
        if head==end or head.next == end:
            return head

        partion = head.next
        tmp_head = ListNode(-1)
        p, tp = partion, tmp_head
        while(p.next!=end):
            if p.next.val < partion.val:
                tp.next = p.next
                tp = tp.next
                p.next = p.next.next
            else:
                p = p.next
        tp.next = head.next
        head.next = tmp_head.next
        self.quick_sort(head, partion)
        self.quick_sort(partion, end)
        return head.next

class Dijkstra:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        graph = collections.defaultdict(list)
        for i, (x, y) in enumerate(edges):
            graph[x].append((succProb[i], y))
            graph[y].append((succProb[i], x))

        que = [(-1.0, start)]
        prob = [0.0] * n
        prob[start] = 1.0

        while que:
            pr, node = heapq.heappop(que)
            pr = -pr
            if pr < prob[node]:
                continue
            for prNext, nodeNext in graph[node]:
                if prob[nodeNext] < prob[node] * prNext:
                    prob[nodeNext] = prob[node] * prNext
                    heapq.heappush(que, (-prob[nodeNext], nodeNext))

        return prob[end]

class MinWindow:
    # 最小覆盖子串
    def minWindow(self, s: str, t: str) -> str:
        need=collections.defaultdict(int)
        for c in t:
            need[c]+=1
        needCnt=len(t)
        i=0
        res=(0,float('inf'))
        for j,c in enumerate(s):
            if need[c]>0:
                needCnt-=1
            need[c]-=1
            if needCnt==0:       #步骤一：滑动窗口包含了所有T元素
                while True:      #步骤二：增加i，排除多余元素
                    c=s[i] 
                    if need[c]==0:
                        break
                    need[c]+=1
                    i+=1
                if j-i<res[1]-res[0]:   #记录结果
                    res=(i,j)
                need[s[i]]+=1  #步骤三：i增加一个位置，寻找新的满足条件滑动窗口
                needCnt+=1
                i+=1
        return '' if res[1]>len(s) else s[res[0]:res[1]+1]    #如果res始终没被更新过，代表无满足条件的结果

class MaximalSquare:
    def maximalSquare(self, matrix):
        m, n = len(matrix), len(matrix[0])
        dp = [ [0]*n for _ in range(m) ]
        for i in range(m):
            if matrix[i][0]=="1":
                dp[i][0] = 1
        for j in range(n):
            if matrix[0][j] == "1":
                dp[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] != "1":
                    dp[i][j] = 0
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
        return max([max(_) for _ in dp])**2

class Isvalid:
    def isvalid(self, s):
        c_dic = {"(":")", "{":"}", "[":"]", "?":"?"}
        stack = ["?"]
        for c in s:
            if c in c_dic:
                stack.append(c)
            else:
                _c = stack.pop(-1)
                if _c != c_dic[c]:
                    return False
        return len(stack)==1

class CombinationSum:
    # 组合总和
    def combinationSum(self, nums, target):
        self.ans = []
        n = len(nums)
        def dfs(nums, begin, size, path, target):
            if target==0:
                self.ans.append(path[:])
                return
            for i in range(begin, size):
                if target<nums[i]: 
                    break
                dfs(nums, i, size, path+[nums[i]], target-nums[i])
        nums.sort()
        path = []
        dfs(nums, 0, n, path, target)
        return self.ans

class Viterbi:
    def viterbi(self, obs, states, start_prob, trans_prob, emission_prob):
        T = len(obs)        # 总的序列长度
        N = len(states)     # 总的状态数

        # 初始化Viterbi矩阵和路径矩阵
        viterbi_mat = np.zeros((N, T))
        path_mat = np.zeros((N, T), dtype=int)

        # 初始状态概率
        viterbi_mat[:, 0] = start_prob * emission_prob[:, obs[0]]
        path_mat[:, 0] = 0

        for t in range(1, T):
            for s in range(N):
                prob = viterbi_mat[:, t-1] * trans_prob[:, s] * emission_prob[s, obs[t]]
                viterbi_mat[s, t] = np.max(prob)    # 表示当前状态为s时的最大概率值
                path_mat[s,t] = np.argmax(prob)     # 表示的是由哪个位置(上一个状态)到当前位置状态为s概率最大
        ## 转化成矩阵形式
        # for t in range(1, T):
        #     prob = viterbi_mat[:, t-1][:,None] + trans_prob + emission_prob[obs[t]] 
        #     viterbi_mat[t] = prob.max(0)
        #     path_mat[t] = prob.argmax(0)

        # 回溯最优路径
        best_path = [np.argmax(viterbi_mat[:, -1])]
        for t in range(T-1, 0, -1):
            best_path.insert(0, path_mat[best_path[0], t])

        return best_path

class FindPeakElement:
    def findPeakElement(self, nums):
        n = len(nums)
        def get(inx):
            if inx==-1 or inx>=n:
                return float("-inf")
            return nums[inx]

        left, rigth = 0, n-1
        ans = -1
        while left<=right:
            mid = (left+right)//2
            if get(mid)>get(mid-1) and get(mid)>get(mid+1):
                ans = mid
                break
            if get(mid)<get(mid+1):
                left = mid+1
            else:
                right= mid-1
        return ans

class BackToOrigin:
    # 圆环回原点问题: 圆环上有10个点，编号为0~9。从0点出发，每次可以逆时针和顺时针走一步，问走n步回到0点共有多少种走法。
    # 走n步到0的方案数=走n-1步到1的方案数+走n-1步到9的方案数。
    def backToOrigin(self, n):
        length = 10 # 0-9 总共10个数
        dp = [ [0 for i in range(length)] for j in range(n) ] #dp[i][j] 表示走i步走到j的方案数
        dp[0][0] = 1

        for i in range(1, n+1):
            for j in range(length):
                dp[i][j] = dp[i-1][(j-1+length)%length] + dp[i-1][(j+1)%length]

        return dp[n][0]

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def merge(head1: ListNode, head2: ListNode) -> ListNode:
            dummyHead = ListNode(0)
            temp, temp1, temp2 = dummyHead, head1, head2
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return dummyHead.next
        
        if not head:
            return head
        
        length = 0
        node = head
        while node:
            length += 1
            node = node.next
        
        dummyHead = ListNode(0, head)
        subLength = 1
        while subLength < length:
            prev, curr = dummyHead, dummyHead.next
            while curr:
                head1 = curr
                for i in range(1, subLength):
                    if curr.next:
                        curr = curr.next
                    else:
                        break
                head2 = curr.next
                curr.next = None
                curr = head2
                for i in range(1, subLength):
                    if curr and curr.next:
                        curr = curr.next
                    else:
                        break
                
                succ = None
                if curr:
                    succ = curr.next
                    curr.next = None
                
                merged = merge(head1, head2)
                prev.next = merged
                while prev.next:
                    prev = prev.next
                curr = succ
            subLength <<= 1
        
        return dummyHead.next

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def merge(left, right):
            dummy = ListNode(-1)
            cur = dummy
            while left and right:
                if left.val <= right.val:
                    cur.next = left
                    left = left.next
                else:
                    cur.next = right
                    right = right.next
                cur = cur.next
            if left:
                cur.next = left
            if right:
                cur.next = right
            return dummy.next
        def mergesort(head):
            if not head or not head.next:
                return head
            slow, fast = head, head.next
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            left_head, right_head = head, slow.next
            slow.next = None
            return merge(mergesort(left_head), mergesort(right_head))
        return mergesort(head)

class Solution:
    def diameterOfBinaryTree(self, root):
        max_path = 0
        def dfs(root):
            if not root:
                return 0
            left_depth = dfs(root.left)
            right_depth = dfs(root.right)
            max_path = max(left_depth + right_depth + 1, max_path)
            return max(left_depth, right_depth) + 1
        dfs(root)
        return max_path-1

class Solution:
    # 单词查找
    def exist(self, board, word):
        self.m, self.n = len(board), len(board[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def dfs(i, j, k, visited):
            if i<0 or i>=self.m or j<0 or j>=self.n or (i,j) in visited:
                return False
            if board[i][j]!=word[k]:
                return False
            if k==len(word)-1:
                return True

            visited.add((i, j))
            for dx, dy in directions:
                n_x, n_y = i+dx, j+dy
                if dfs(n_x, n_y, k+1, visited):
                    return True
            visited.remove((i,j))
            return False

        visited = set()
        for i in range(self.m):
            for j in range(self.n):
                if dfs(i, j, 0, visited):
                    return True
        return False

class DeleteDuplicates:
    # 删除链表中的重复元素
    def deleteDuplicates2(self, head):
        if not head or not head.next:
            return head

        dummynode = ListNode(val=-1)
        dummynode.next = head
        pre, cur = dummynode, head
        while cur:
            nxt = cur.next
            flag = False
            while nxt and nxt.val==cur.val:
                flag = True
                nxt = nxt.next

            if flag:
                pre.next = nxt
            else:
                pre = cur
            cur = nxt
        return dummynode.next

    def deleteDuplicates(self, head):
        if not head or not head.next:
            return head

        dummynode = ListNode(val=-1, next=head)
        cur = head
        while cur:
            nxt = cur.next
            while nxt.val==cur.val:
                nxt = nxt.next
            cur.next = nxt
            cur = cur.next
        return dummynode.next
        
class GenerateParenthesis:
    # 括号生成
    def generateParenthesis(self, n):
        if n<=0:
            return ''
        elif n==1:
            return '()'
        res = set()
        for i in self.generateParenthesis(n-1):
            for j in range(len(i)):
                res.add(i[0:j] + '()' + i[j:])
        return list(res)

class InorderTraversal:
    def inorderTraver(self, root):
        if not root:
            return
        self.ans = []
        stack = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            self.ans.append(root.val)
            root = root.right
        return self.ans

class dailyTemperatures:
    def dailyTemperatures(self, temperatures):
        n = len(temperatures)
        result = [0]*n
        stack = []
        for i,temperature in enumerate(temperatures):
            while stack and temperatures[stack[-1]]<temperature:
                pre_inx = stack.pop()
                result[pre_inx] = i-pre_inx
            stack.append(i)

        return result

class Subset:
    # 子集
    def subset(self, nums):
        # 使用回溯
        size = len(nums)
        sets = [[]]
        for n in nums:
            for j in range(len(sets)):
                sets.append(set[j]+n)
        return sets

class KthLargest:
    def Kthlargest(self, root, k):
        self.res = None
        self.k = k
        def dfs(root):
            if not root:
                return
            dfs(root.right)
            self.k -= 1
            if self.k==0:
                self.res = root.val
                return 
            dfs(root.left)
        dfs(root)
        return self.res

INT_MAX = 2 ** 31 - 1
INT_MIN = -2 ** 31
class Automaton:
    def __init__(self):
        self.state = 'start'
        self.sign = 1
        self.ans = 0
        self.table = {
            'start': ['start', 'signed', 'in_number', 'end'],
            'signed': ['end', 'end', 'in_number', 'end'],
            'in_number': ['end', 'end', 'in_number', 'end'],
            'end': ['end', 'end', 'end', 'end'],
        }

    def get_col(self, c):
        if c.isspace():
            return 0
        if c == '+' or c == '-':
            return 1
        if c.isdigit():
            return 2
        return 3

    def get(self, c):
        self.state = self.table[self.state][self.get_col(c)]
        if self.state == 'in_number':
            self.ans = self.ans * 10 + int(c)
            self.ans = min(self.ans, INT_MAX) if self.sign == 1 else min(self.ans, -INT_MIN)
        elif self.state == 'signed':
            self.sign = 1 if c == '+' else -1

class Solution:
    def myAtoi(self, str: str) -> int:
        automaton = Automaton()
        for c in str:
            automaton.get(c)
        return automaton.sign * automaton.ans

class KMP:
    def __init__(self, pattern):
        self.pattern = pattern
        self.next_list = self._build_next(pattern)

    def _build_next(self, pattern):
        self.next_list = [0]
        prefix_len = 0
        i = 1
        while i < len(self.pattern):
            if self.pattern[i] == self.pattern[prefix_len]:
                prefix_len += 1
                self.next_list.append(prefix_len)
                i += 1
            else:
                if prefix_len == 0:
                    self.next_list.append(0)
                    i += 1
                else:
                    prefix_len = next_list[prefix_len-1]

    def search(self, source):
        m = len(source)
        n = len(self.pattern)

        i,j = 0, 0
        while i<m:
            if source[i]==self.pattern[j]:
                i+=1
                j+=1
            elif j>0:
                j = self.next_list[j-1]
            else:
                i += 1  # 第一个字符就没有匹配到

            if j==n:
                return i-j

        return -1

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        newHead = head.next
        head.next = self.swapPairs(newHead.next)
        newHead.next = head
        return newHead

    def swapPairs(self, head):
        dummynode = ListNode(-1)
        dummynode.next = head
        tmp = dummynode
        while tmp.next and tmp.next.next:
            node1 = tmp.next
            node2 = tmp.next.next
            tmp.next = node2
            node1.next = node2.next
            node2.next = node1
            tmp = node1

        return dummynode.next

class Solution:
    # 二叉树展开为链表
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        stack = [root]
        pre = None

        while stack:
            cur = stack.pop()
            if pre:
                pre.left = None
                pre.right= cur
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
            pre = cur

class Solution:
    # IP 复原
    def restoreIpAddresses(self, s: str) -> List[str]:
        size = len(s)
        if size>12 or size<4:
            return []
        path = []
        self.res = []
        self.dfs(s, size, 0, 0, path)
        return self.res

    def dfs(self, s, size, split_times, begin, path):
        if begin==size:
            if split_times == 4:
                self.res.append('.'.join(path))
            return
        # 如果剩下的部分长度小于可分配最小长度或者大于可分配最大长度，则无效;
        if size-begin<(4-split_times) or size-begin>3*(4-split_times):
            return
        for i in range(3):
            if begin+i>size:
                break
            ip_segment = self.judge_if_ip_segment(s, begin, begin+i)
            if ip_segment!=-1:
                path.append(str(ip_segment))
                self.dfs(s, size, split_times+1, begin+i+1, path)
                path.pop()

    def judge_if_ip_segment(self, left, right):
        size = right-left+1
        if size>1 and s[left]=='0': # 以'0'开头不合法
            return -1
        res = int(s[left:right+1])
        if res>255:
            return -1
        return res

# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    # 复制带随机指针的链表
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        
        # A->A'->B->B'
        p = head
        while p:
            new_node = Node(p.val)
            new_node.next = p.next
            p.next = new_node
            p = new_node.next
        
        # copy random
        p = head
        while p:
            if p.random:
                p.next.random = p.random.next
            p = p.next.next
        
        # 拆分链表
        p = head
        dummy = Node(-1)
        cur = dummy
        while p:
            cur.next = p.next
            cur = cur.next
            p.next = cur.next
            p = p.next
            
        return dummy.next

    def copyRandomList(self, head):
        if not head:
            return head
        cur = head
        nodemap = dict()
        while cur:
            nodemap[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        while cur is not None:
            nxt = nodemap.get(cur.next, None)
            nodemap[cur].next = nxt
            random = nodemap.get(cur.random, None)
            nodemap[cur].random = random
            cur = cur.next
        return nodemap[head]

class SumNumbers:
    def sumNumbers(self, root):
        self.sum = 0
        def dfs(root, num):
            if not root:
                return 0
            num = 10*num + root.val
            if not root.left and not root.right:
                self.sum += num

            dfs(root.left, num)
            dfs(root.right, num)

        dfs(root, 0)
        return self.sum

class WidthOfBinaryTree:
    # 二叉树最大宽度
    def widthOfBinaryTree(self, root):
        self.width = 1
        stack = []
        if not root:
            return 0
        stack.append((1, root))
        while stack:
            width = stack[-1][0] - stack[0][0] + 1
            self.width = max(width, self.width)

            size = len(stack)
            for i in range(size):
                (inx,node) = stack.pop(0)
                if node.left:
                    stack.append((node.left, inx*2))
                if node.right:
                    stack.append((node.right, inx*2+1))
        return self.width

class NextGreaterElement:
    def nextGreaterElement(self, n):
        nums = list(str(n)) 
        i = len(nums)-2
        while i>=0 and nums[i]>=nums[i+1]:
            i -= 1
        if i<0:
            return -1

        j = len(nums)-1
        while j>=0 and nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
        nums[i+1:] = nums[i+1:][::-1]
        ans = int(''.join(nums))
        return ans if ans < 2 ** 31 else -1

class OddEvenList:
    # 奇偶链表
    def oddEvenList(self, head):
        if not head and not head.next:
            return head
        even_head = head.next
        odd, even = head, even_head
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = even_head
        return head

class rotateRight:
    # 旋转链表
    def rotateRight(self, head, k):
        if not head:
            return head

        size = 0
        cur = head
        while cur:
            cur = cur.next
            size += 1
        k = k%size
        if k==0:
            return head
        slow, fast = head, head.next
        for i in range(k):
            fast = fast.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next
        newhead = slow.next
        slow.next = None
        fast.next = head
        return newhead

class Solution:
    # 最佳买卖股票时机
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if not prices:
            return 0

        n = len(prices)
        k = min(k, n // 2)
        buy = [[0] * (k + 1) for _ in range(n)]
        sell = [[0] * (k + 1) for _ in range(n)]

        buy[0][0], sell[0][0] = -prices[0], 0
        for i in range(1, k + 1):
            buy[0][i] = sell[0][i] = float("-inf")

        for i in range(1, n):
            buy[i][0] = max(buy[i - 1][0], sell[i - 1][0] - prices[i])
            for j in range(1, k + 1):
                buy[i][j] = max(buy[i - 1][j], sell[i - 1][j] - prices[i])
                sell[i][j] = max(sell[i - 1][j], buy[i - 1][j - 1] + prices[i]);  

        return max(sell[n - 1])

    def maxProfit(self, k, prices):
        if not prices:
            return 0
        n = len(prices)
        k = min(k, n//2)
        buy = [ -prices[0] ] *(k+1)
        sell = [0] *(k+1)

        for i in range(1, n):
            for j in range(1, k+1):
                buy[j] = max(buy[j], sell[j-1]-prices[i])
                sell[j]= max(sell[j], buy[j]+prices[i])
        return sell[k]

class ShortestSubarray:
    # 和最少为k的最短数组: 先计算前缀和，然后比较
    def shortestSubarray(self, nums, k):
        from itertools import accumulate
        ans = float("inf")
        s = list(accumulate(nums, initial=0))
        q = deque()
        for i, cur_s in enumerate(s):
            while q and cur_s-s[q[0]]>=k:
                ans = min(ans, i-q.popleft())
            while q and s[q[-1]]>=cur_s:
                q.pop()

            q.append(i)
        return ans if ans<inf else -1

class Solution:
    # 将有序数组转换为二叉搜索树
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def helper(left, right):
            if left > right:
                return None

            # 总是选择中间位置右边的数字作为根节点
            mid = (left + right + 1) // 2

            root = TreeNode(nums[mid])
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)
            return root

        return helper(0, len(nums) - 1)

class Solution:
    def __init__(self, w):
        # self.pre_sum = [w[0]]
        # for _w in w[1:]:
        #     self.pre_sum.append(self.pre_sum[-1]+_w)
        self.pre_sum = list(accumulate(w))
        self.total = sum(w)

    def selectindex(self):
        x = random.randint(1, self.total)
        # left, right = 0, len(self.pre_sum)-1
        # while left + 1 < right:
        #     mid = (left + right) // 2
        #     if x >= self.preSum[mid]:
        #         left = mid
        #     else:
        #         right = mid
        # if x < self.preSum[left]:
        #     return left
        # return right
        return bisect_left(self.pre_sum, x)

class Solution:
    # 蓄水池抽样算法:蓄水池抽样算法用于从一个数据流中随机选择k个元素，且保证每个元素被选择的概率相等;
    def reservoir_sampling(self, stream, k):
        reservoir = []
        n = 0
        for item in stream:
            n += 1
            if len(reservoir)<k:
                reservoir.append(item)
            else:
                s = random.randint(0, n-1)
                if s<k:
                    reservoir[s] = item
        return reservoir

class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        self.ans = [ [0]*n for _ in range(n) ]
        left, right, top, buttom = 0, n-1, 0, n-1
        i = 1
        while i <= (n*n):
            if left>right:break
            for _ in range(left, right+1):
                self.ans[top][_] = i
                i += 1
            top += 1
            if top>buttom:break
            for _ in range(top, buttom+1):
                self.ans[_][right] = i
                i += 1
            right -= 1
            if right<left:break
            for _ in range(right, left-1, -1):
                self.ans[buttom][_] = i
                i += 1
            buttom -= 1
            if buttom<top:break
            for _ in range(buttom, top-1, -1):
                self.ans[_][left] = i
                i += 1
            left += 1

        return self.ans

class Solution:
    # 单词切分
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n+1)
        dp[0] = True

        for i in range(n):
            for j in range(i+1, n+1):
                if dp[i] and (s[i:j] in wordDict):
                    dp[j] = True

        return dp[-1] 

class Solution:
    # 旋转数组最小值
    def minArray(self, nums):
        left, right = 0, len(nums)-1
        while left<right:
            mid = (left+right)//2
            if nums[mid]>nums[right]:
                left = mid+1
            elif nums[mid]<nums[left]:
                right = mid
            else:
                right -= 1
        return nums[left]

class Solution:
    # 组合总和II
    def combinationSum2(self, candidates, target):
        def dfs(begin, path, residue):
            if residue == 0:
                res.append(path[:])
                return

            for i in range(begin, size):
                if candidates[i] > residue: # 剪枝
                    break
                if i>begin and candidates[i-1] == candidates[i]:
                    continue

                path.append(candidates[i])
                dfs(i+1, path, residue-candidates[i])
                path.pop()

        size = len(candidates)
        if size==0:
            return []
        candidates.sort()
        res, path = [], []
        dfs(0, path, target)
        return res

class Solution:
    # 字符串数字 删除k个数后最小
    def removeKdigits(self, num: str, k: int) -> str:
        numStack = []
        
        # 构建单调递增的数字串
        for digit in num:
            while k and numStack and numStack[-1] > digit:
                numStack.pop()
                k -= 1
        
            numStack.append(digit)
        
        # 如果 K > 0，删除末尾的 K 个字符
        finalStack = numStack[:-k] if k else numStack
        
        # 抹去前导零
        return "".join(finalStack).lstrip('0') or "0"

class Find132pattern:
    # 132模式
    def find132pattern(self, nums: List[int]) -> bool:
        stack = []
        k = -(10 ** 9 + 7)
        for i in range(len(nums) - 1,-1,-1):
            # 存在 nums[i]<k，而 k 的存在必定是满足有j，是的 nums[j]>k的；
            if nums[i] < k:
                return True
            while stack and stack[-1] < nums[i]:
                # 在不满足单调递减时 k 才有值，也就是说在 k 有值的情况下，肯定满足存在j 且 nums[j] > k;
                k = max(k,stack.pop())
            stack.append(nums[i])
        return False

class Solution:
    # 山脉数组中查找
    def findInMountainArray(self, target, mountain_array):
        l, r = 0, len(mountain_array)-1
        while l<=r:
            mid = (l+r)//2
            if mountain_array[mid] < mountain_array[mid+1]:
                l = mid+1
            else:
                r = mid

        peak = l
        index = self.binary_search(target, mountain_array, 0, peak)
        if index != -1:
            return index
        return self.binary_search(target, mountain_array, peak, len(mountain_array)-1, False)

    def binary_search(self, target, array, left, right, ascend=True):
        l, r = left, right
        while l<=r:
            mid = (l+r)//2
            if array[mid] == target:
                return mid
            if array[mid]>target:
                if ascend:
                    r = mid-1
                else:
                    l = mid+1
            else:
                if ascend:
                    l = mid+1
                else:
                    r = mid-1

        return -1

class Solution:
    # 字符片段切割
    def partionLabels(self, s):
        last = [0]*26
        for i,c in enumerate(s):
            last[ord(c)-ord('a')] = i

        start, end = 0, 0
        partion = []
        for i,c in enumerate(s):
            end = max(end, last[ord(c)-ord('a')])
            if i==end:
                partion.append(end-start+1)
                start = end+1
        return partion

class Solution:
    # 判断是否是顺子
    def isStraight(self, nums):
        joker = 0
        nums.sort()
        for i in range(4):
            if nums[i]==0:
                joker += 1
            elif nums[i]==nums[i+1]:
                return False
        return nums[4]-nums[joker] < 5

class nextGreaterElements:
    # 循环数组下一个更大的数
    def nextGreaterElements(self, nums):
        n = len(nums)
        ret = [-1] * n
        stk = list()

        for i in range(n * 2 - 1):
            while stk and nums[stk[-1]] < nums[i % n]:
                ret[stk.pop()] = nums[i % n]
            stk.append(i % n)

        return ret

class MoveZeros:
    # 移动0值到最后 in-places
    def moveZeros(self, nums):
        left, right = 0, 0
        while right<len(nums):
            if nums[right]!=0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1

class Solution:
    # 丑数
    def nthUglyNumber(self, n: int) -> int:
        dp = [0]*(n+1)
        dp[0] = 1
        dp[1] = 1
        p2, p3, p5 = 1, 1, 1

        for i in range(2, n+1):
            dp[i] = min(2*dp[p2], 3*dp[p3], 5*dp[p5])

            # 这里都要更新
            if 2*dp[p2] == dp[i]:
                p2 += 1
            if 3*dp[p3] == dp[i]:
                p3 += 1
            if 5*dp[p5] == dp[i]:
                p5 += 1

        return dp[n]

class Solution:
    # 字符串解码
    def decodeString(self, s: str) -> str:
        stack = []
        
        for c in s:
            if c == ']':
                tmp = ''
                while stack:
                    _c = stack.pop()
                    if _c=='[':
                        break
                    else:
                        tmp = _c + tmp
                n = ''
                while stack and stack[-1].isdigit():
                    _n = stack.pop()
                    n = _n + n
                n = int(n)
                stack.append(tmp * n)
            else:
                stack.append(c)
        
        return ''.join(stack)

class simplifyPath:
    # 简化路径字符串
    def simplifyPath(self, path):
        stack = []
        items = path.split('/')
        for item in items:
            if not item:
                continue
            if not stack and item=="..":
                continue
            elif item==".":
                continue
            elif item=="..":
                stack.pop()
            else:
                stack.append(item)

        return "/" + '/'.join(stack)

class Partition:
    # 分隔链表
    def partition(self, head, x):
        smalldummynode = ListNode(-1)
        largedummynode = ListNode(-1)
        small, large = smalldummynode, largedummynode
        while head is not None:
            if head.val>=x:
                large.next = head
                large = large.next
            else:
                small.next = head
                small = small.next
            head = head.next

        large.next = None
        small.next = largedummynode.next
        return smalldummynode.next

class FourSum:
    # 四数之和
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        quadruplets = list()
        if not nums or len(nums) < 4:
            return quadruplets
        
        nums.sort()
        length = len(nums)
        for i in range(length - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                break
            if nums[i] + nums[length - 3] + nums[length - 2] + nums[length - 1] < target:
                continue
            for j in range(i + 1, length - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                if nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target:
                    break
                if nums[i] + nums[j] + nums[length - 2] + nums[length - 1] < target:
                    continue
                left, right = j + 1, length - 1
                while left < right:
                    total = nums[i] + nums[j] + nums[left] + nums[right]
                    if total == target:
                        quadruplets.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        right -= 1
                    elif total < target:
                        left += 1
                    else:
                        right -= 1
        
        return quadruplets

class MirrorTree:
    def mirrorTree(self, root):
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.mirrorTree(root.left)
        self.mirrorTree(root.right)
        return root

class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = collections.deque()


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        n = len(self.queue)
        self.queue.append(x)
        for _ in range(n):
            self.queue.append(self.queue.popleft())


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.queue.popleft()


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.queue[0]


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return not self.queue

class Solution:
    # 回文子串个数
    def countSubstrings(self, s):
        n = len(s)
        ans = 0
        for i in range(2*n-1):
            # fuck coding: 
            l, r = i//2, i//2+i%2
            while (l>=0 and r<n and s[l]==s[r]):
                l -= 1
                r += 1
                ans += 1

        return ans

class SearchRange:
    def searchRange(self, nums, target):
        def binary_search(nums, target):
            l, r = 0, len(nums)-1
            while l<=r:
                mid = l+(r-l)//2
                if nums[mid] >= target:
                    r = mid-1
                else:
                    l = mid+1
            return l
        # left = bisect_left(nums, target)
        # right = bisect_left(nums, target+1)-1
        left = binary_search(nums, target)
        right= binary_search(nums, target+1)-1
        if left==len(nums) or nums[left]!=target:
            return [-1, -1]
        else:
            return [left, right]

class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        memo = {}
        def dp(k, n):
            if (k, n) not in memo:
                if n == 0:
                    ans = 0
                elif k == 1:
                    ans = n
                else:
                    lo, hi = 1, n
                    # keep a gap of 2 x values to manually check later
                    while lo + 1 < hi:
                        x = (lo + hi) // 2
                        t1 = dp(k - 1, x - 1)
                        t2 = dp(k, n - x)

                        if t1 < t2:
                            lo = x
                        elif t1 > t2:
                            hi = x
                        else:
                            lo = hi = x

                    ans = 1 + min(max(dp(k - 1, x - 1), dp(k, n - x))
                                  for x in (lo, hi))

                memo[k, n] = ans
            return memo[k, n]

        return dp(k, n)

class ReorganizeString:
    from collections import Counter
    def reorganizeString(self, s):
        if len(s)<2:
            return s
        n = len(s)
        counters = Counter(s)
        if counters.most_common(1)[0][-1] > (n+1)//2:
            return ''

        queue = [(-x[1], x[0]) for x in counts.items()]
        heapq.heapify(queue)
        ans = list()

        while len(queue)>1:
            _, letter1 = heapq.heappop(queue)
            _, letter2 = heapq.heappop(queue)
            ans.extend([letter1, letter2])
            counts[letter1] -= 1
            counts[letter2] -= 1
            if counts[letter1] > 0:
                heapq.heappush(queue, (-counts[letter1], letter1))
            if counts[letter2] > 0:
                heapq.heappush(queue, (-counts[letter2], letter2))

        if queue:
            ans.append(queue[0][1])

        return "".join(ans)

class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        ans = 0

        def dfs(x: int, y: int) -> bool:
            if x < 0 or y < 0 or x >= m or y >= n:
                return False
            if grid[x][y] != 0:
                return True
            
            grid[x][y] = -1
            ret1, ret2, ret3, ret4 = dfs(x - 1, y), dfs(x + 1, y), dfs(x, y - 1), dfs(x, y + 1)
            return ret1 and ret2 and ret3 and ret4
            # return dfs(x-1, y) and dfs(x+1, y) and dfs(x, y-1) and dfs(x, y+1) # 这种写法有问题
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and dfs(i, j):
                    ans += 1
        
        return ans

class topKFrequent:
    # 最高频的k个数字
    def topKFrequent(self, nums, k):
        cnt = {}
        for num in nums:
            cnt[num] = cnt.get(num, 0)+1

        candidates = list()
        for u,v in cnt.items():
            heapq.heappush(candidates, (v, u))
            while len(candidates)>k:
                heapq.heappop(candidates)

        ans = [_[1] for _ in candidates]
        return ans

class NumWays:
    # 青蛙跳台阶
    def numWays(self, n):
        if n==0:
            return 1
        if n==1:
            return 1
        # f(n) = f(n-1) + f(n-2)
        a, b, c = 1, 1, 0
        for i in range(2, n+1):
            c = a+b
            a, b = b, c
        return c

class MinNumber:
    # 把数组排成最小数
    def minNumber(self, nums):
        def sort_rule(x, y):
            a, b = x+y, y+x
            if a>b:
                return 1
            elif a<b:
                return -1
            else:
                return 0
        strs = [str(num) for num in nums]
        strs.sort(key=functools.cmp_to_key(sort_rule))
        return ''.join(strs)

    def minNumber(self, nums: List[int]) -> str:
        def quick_sort(l , r):
            if l >= r: return
            i, j = l, r
            while i < j:
                while strs[j] + strs[l] >= strs[l] + strs[j] and i < j:
                    j -= 1
                while strs[i] + strs[l] <= strs[l] + strs[i] and i < j: 
                    i += 1
                strs[i], strs[j] = strs[j], strs[i]

            strs[i], strs[l] = strs[l], strs[i]
            quick_sort(l, i - 1)
            quick_sort(i + 1, r)
        
        strs = [str(num) for num in nums]
        quick_sort(0, len(strs) - 1)
        return ''.join(strs)

class RecoverTree:
    # 恢复二叉搜索树
    def recoverTree(self, root):
        nodes = []
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            nodes.append(root)
            dfs(root.right)
        dfs(root)
        x, y = None, None
        pre = nodes[0]
        for i in range(1, len(nodes)):
            if pre.val > nodes[i].val:
                y = nodes[i]
                if not x:
                    x = pre
            pre = nodes[i]
        if x and y:
            x.val, y.val = y.val, x.val

    def recoverTree(self, root):
        self.x, self.y, self.pre = None, None, None
        def dfs(root):
            if not root:
                return
            dfs(root.left)

            if self.pre and self.pre.val>root.val:
                self.y = root
                if not self.x:
                    self.x = self.pre
            self.pre = root

            dfs(root.right)

        dfs(root)
        if self.x and self.y:
            self.x.val,self.y.val = self.y.val,self.x.val

class IsPalindrome:
    def isPalindrome(self, s: str) -> bool:
        sgood = "".join([c.lower() for c in s if c.isalnum()])
        return sgood == sgood[::-1]

class Exchange:
    def exchange(self, nums):
        i, j = 0, len(nums)-1
        while i<j:
            while i<j and nums[i]%2 == 1:
                i += 1
            while i<j and nums[j]%2 == 0:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
        return nums

class CheckSubarraySum:
    def checkSubarraySum(self, nums, k):
        n = len(nums)
        if n<2:
            return False
        mp = {0:-1}
        reminder = 0
        for i in range(n):
            reminder = (reminder+nums[i]) % k
            if reminder in mp:
                pre_inx = mp[reminder]
                if i-pre_inx>=2:
                    return True
            else:
                mp[reminder] = i
        return False

class lengthOfLongestSubstring:
    # 最长不含重复字符的子串
    def lengthOfLongestSubstring(self, s):
        window = []
        ret = 0
        for c in s:
            while c in set(w):
                window.pop(0)
            window.append(c)
            ret = max(ret, len(window))
        return ret

    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = {}
        res = tmp = 0
        for j in range(len(s)):
            i = dic.get(s[j], -1) # 获取索引 i
            dic[s[j]] = j # 更新哈希表
            tmp = tmp + 1 if tmp < j - i else j - i # dp[j - 1] -> dp[j]
            res = max(res, tmp) # max(dp[j - 1], dp[j])
        return res

class findNumberIn2DArray:
    def findNumberIn2DArray(self, matrix, target):
        i, j = 0, len(matrix[0])-1
        while i<len(matrix)-1 and j>=0:
            if matrix[i][j] == target:
                return True
            if matrix[i][j]<target:
                i += 1
            else:
                j -= 1
        return False

class Solution:
    # 双栈排序
    def stackSort(self, stk):
        tmp = []
        while stk:
            peak = stk.pop()
            while tmp and tmp[-1] > peak:
                t = tmp.pop()
                stk.append(t)
            tmp.append(peak)
        return tmp

class Trie:
    def __init__(self):
        self.children = [None]*26
        self.isEnd = False

    def insert(self, word):
        node = self
        for w in word:
            w = ord(w)-ord("a")
            if not node.children[w]:
                node.children[w] = Trie()
            node = node.children[w]
        node.isEnd = True

    def search_prefix(self, prefix):
        node = self
        for w in prefix:
            w = ord(w) - ord("a")
            if not node.children[w]:
                return None
            node = node.children[w]
        return node

    def search(self, word):
        node = self.search_prefix(word)
        return node is not None and node.isEnd
        
    def start_with(self, word):
        node = self.search_prefix(word)
        return node is not None

class IsMatch:
    # 正则表达式匹配 .*
    def isMatch(self, s, p):
        m, n = len(s), len(p)
        dp = [ [False]*(n+1) for _ in range(m+1) ]

        dp[0][0] = True
        for i in range(m+1):
            for j in range(1, n+1):
                if p[j-1] != "*":
                    if (i>=1 and (p[j-1]==s[i-1] or p[j-1]==".")):
                        dp[i][j] = dp[i-1][j-1]
                else:
                    if j>=2:
                        dp[i][j] |= dp[i][j-2]
                    if (i>=1 and j>=2 and (s[i-1]==p[j-2] or p[j-2]=='.')):
                        dp[i][j] |= dp[i-1][j]

        return dp[m][n]

    def isMatch_v2(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        def matches(i, j) -> bool:
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        f = [[False] * (n + 1) for _ in range(m + 1)]
        f[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    f[i][j] |= f[i][j - 2]
                    if matches(i, j - 1):
                        f[i][j] |= f[i - 1][j]
                else:
                    if matches(i, j):
                        f[i][j] |= f[i - 1][j - 1]
        return f[m][n]

class IsMatch:
    # 正则表达式匹配 ?*
    def isMatch(self, s, p):
        m, n = len(s), len(p)

        dp = [ [False]*(n+1) for _ in range(m+1)]
        dp[0][0] = True
        for i in range(1, n+1):
            # 只有前面全是*时才会为True, 否则break
            if p[i-1] == "*":
                dp[0][i] = True
            else:
                break

        for i in range(1, m+1):
            for j in range(1, n+1):
                if p[j-1] == "*":
                    dp[i][j] = dp[i][j-1] | dp[i-1][j]
                elif p[j-1] == "?" or s[i-1]==p[j-1]:
                    dp[i][j] = dp[i-1][j-1]

        return dp[m][n]

class FreqStack:
    def __init__(self):
        self.freq = defaultdict(int)
        self.group= defaultdict(list)
        self.max_freq = 0
    
    def push(self, val):
        self.freq[val] += 1
        cnt = self.freq[val]
        self.group[cnt].append(val)
        self.max_freq = self.max_freq if self.max_freq>cnt else cnt

    def pop(self):
        val = self.group[self.max_freq].pop()
        self.freq[val] -= 1
        if len(self.group[self.max_freq])<=0:
            self.max_freq -= 1

        return val

class RandomPick:
    def __init__(self, nums):
        self.num_inx = defaultdict(list)
        for i,num in enumerate(nums):
            self.[num].append(i)

    def pick(self, num):
        return random.choice(self.num_inx[num])

    # 如果nums 以文件的方式存储，文件大小远远大于内存大小，则上面的方法无法使用;
    def __init__(self, nums):
        self.nums = nums
    
    def pick(self, num):
        ans = cnt = 0
        for i,num in enumerate(self.nums):
            if num==target:
                cnt += 1  # 第cnt 次遇到target
                if randrange(cnt) == 0:
                    ans = i

        return ans

class Jump:
    def jump(self, nums):
        # 贪心算法
        n = len(nums)
        max_pos, end, step = 0, 0, 0
        for i in range(n-1):
            if max_pos >= i:
                max_pos = max(max_pos, i+nums[i])
                if i==end:  # 当 i 到达 end 时, 表示需要跳跃
                    end = max_pos
                    step += 1

        return step

    def jump(self,nums):
        # 贪心算法
        n = len(nums)
        if n==1:
            return 0
        steps, cur_dis, next_dis = 0, 0, 0
        for i in range(n):
            next_dis = max(next_dis, i+nums[i])
            if i == cur_dis:
                cur_dis = next_dis
                steps += 1
                if next_dis >= (n-1):
                    break
        return steps

    def jump_v2(self, nums):
        # 动态规划
        n = len(nums)
        if n == 1:
            return 0;

        dp = [ 1e5 ] * n
        dp[n-1] = 0
        for i in range(n-2, -1, -1):
            jump_largest = i+nums[i]
            if jump_largest >= n:
                dp[i] = 1
                continue
            for j in range(i+1, jump_largest):
                dp[i] = min(dp[i], dp[j])
            dp[i] += 1
        return dp[0]

class Canpartition:
    def canpartition(self, nums):
        size = len(nums)
        if size<=1:
            return False
        num_sum = sum(nums)
        if num_sum & 1:
            return False
        nums.sort()

        target = num_sum // 2
        if nums[-1] > target:
            return False

        dp = [ [False]*(target+1) for _ in range(size) ]
        for i in range(size):
            dp[i][0] = True
        dp[0][nums[0]] = True
        for i in range(1, n):
            num = nums[i]
            for j in range(1, target+1):
                if j>=num:
                    dp[i][j] = dp[i-1][j] | dp[i-1][j-num]
                else:
                    dp[i][j] = dp[i-1][j]

        return dp[size-1][target]

class MinimumTotal:
    # 三角最短路径和
    def minimumTotal(self, triangle):
        n = len(triangle)
        f = [ [0]*n for _ in range(n) ]
        f[0][0] = triangle[0][0]

        for i in range(1, n):
            f[i][0] = f[i-1][0] + triangle[i][0]    # 第一个
            for j in range(1, i):
                f[i][j] = min(f[i-1][j-1], f[i-1][j]) + triangle[i][j]
            f[i][i] = f[i-1][i-1] + triangle[i][i]   # 最后一个
                
        return min(f[-1])

    def minimumTotal(self, triangle):
        memo = triangle[-1]
        n = len(memo)
        for i in range(n-2, -1, -1):
            for j in range(i+1):
                memo[j] = triangle[i][j] + min(memo[j], memo[j+1])
        return memo[0]

class Solution:
    # 链表插入排序
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        dummyHead = ListNode(0)
        dummyHead.next = head
        lastSorted = head
        curr = head.next

        while curr:
            if lastSorted.val <= curr.val:
                lastSorted = lastSorted.next
            else:
                prev = dummyHead
                while prev.next.val <= curr.val:
                    prev = prev.next
                lastSorted.next = curr.next
                curr.next = prev.next
                prev.next = curr
            curr = lastSorted.next
        
        return dummyHead.next

class Solution:
    # 判断是否二分图
    def isBipartite(self, graph:List[List[int]]) -> bool:
        # 广度优先
        n = len(graph)
        uncolor, red, green = 0, 1, 2
        color = [ uncolor ] * n

        for i in range(len(n)):
            if color[i]==uncolor:
                q = collections.deque([i])
                color[i] = red
                while q:
                    node = q.popleft()
                    neighbor_color = green if color[node]==red else red
                    for neighbor in graph[node]:
                        if color[neighbor]==uncolor:
                            q.append(neighbor)
                            color[neighbor]=neighbor_color
                        elif color[neighbor]==neighbor_color:
                            continue
                        else:
                            return False

        return True

    def isBipartite(self, graph):
        # 深度优先
        n = len(graph)
        uncolor, red, green = 0, 1, 2
        color = [uncolor] * n
        valid = True

        def dfs(node, c):
            nonlocal valid
            color[node] = c
            neighbor_color= green if c==red else red
            for neighbor in graph[node]:
                if color[neighbor] == uncolor:
                    dfs(neighbor, neighbor_color)
                    if not valid:
                        return
                elif color[neighbor]!=neighbor_color:
                    valid = False
                    return

        for i in range(n):
            if color[i]==uncolor:
                dfs(i, red)
                if not valid:
                    break
        return valid

class FindMedian:
    def __init__(self):
        self.min_queue = list()
        self.max_queue = list()

    def addNum(self, num):
        min_queue = self.min_queue
        max_queue = self.max_queue
        
        if not min_queue or num <= -min_queue[0]:
            heapq.heappush(min_queue, -num)
            if len(max_queue+1) < len(min_queue):
                heapq.heappush(max_queue, -heapq.heappop(min_queue))
        else:
            heapq.heappush(max_queue, num)
            if len(max_queue) > len(min_queue):
                heapq.heappush(min_queue, -heapq.heappop(max_queue))

    def findMedian(self):
        min_queue = self.min_queue
        max_queue = self.max_queue

        if len(min_queue) > len(max_queue):
            return -min_queue[0]
        else:
            return (-min_queue[0] + max_queue[0]) * 0.5

class sumOfLeftLeaves:
    # 左叶子节点和
    def sumOfLeftLeaves(self, root):
        self.ans = 0
        def dfs(root, isleft):
            if not root:
                return
            if not root.left and not root.right and is_left:
                self.ans += root.val
                return 
            dfs(root.left, True)
            dfs(root.right, False)
        dfs(root, False)

        return self.ans

class Solution:
    def largestRectangleArea(self, heights):
        n = len(heights)
        stack = []
        right_stack = []
        for i in range(n-1, -1, -1):
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            if stack:
                right_stack.append(stack[-1])
            else:
                right_stack.append(n)
            stack.append(i)
        right_stack = right_stack[::-1]

        stack = []
        left_stack = []
        for i in range(n):
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            if stack:
                left_stack.append(stack[-1])
            else:
                left_stack.append(-1)
            stack.append(i)

        max_area = 0
        for i in range(n):
            max_area = max(max_area, heights[i]*(right_stack[i]-left_stack[i]-1))
        return max_area

class LongestCommonPrefix:
    def longestCommonPrefix(self, strs):
        if not strs:
            return ''
        max_size = min([len(s) for s in strs])
        ans = ''
        for i in range(max_size):
            cs = set([s[i] for s in strs])
            if len(cs)>=2:
                break
            ans += cs.pop()

        return ans

class SubarrayDivByK:
    # 和可被K整除的子数组, 前缀和 + 同余
    def subarrayDivByK(self, nums):
        record = {0:1}
        total, ans = 0, 0
        for num in nums:
            total += num
            modulus = total % k
            same = record.get(modulus, 0)
            ans += same
            record[modulus] = same + 1
        return ans

class SortArrayByPartity:
    def sortArrayByPartity(self, nums):
        n = len(nums)
        j = 1
        for i in range(0, n, 2):
            if nums[i]%2 == 1:
                while j<n-1 and nums[j]%2 == 1:
                    j += 2
                nums[i], nums[j] = nums[j], nums[i]

        return nums

class sortColors:
    # 颜色排序
    def sortcolors(self, nums):
        n = len(nums)
        p0, p2 = 0, n-1
        i = 0
        while i<=p2:
            while i<=p2 and nums[i]==2:
                nums[i], nums[p2] = nums[p2], nums[i]
                p2 -= 1
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                p0 += 1
            i += 1

class HammingWeight:
    # 位 1 的个数
    def hammingWeight(self, n):
        ret = 0
        while n:
            n &= n-1
            ret += 1

        return ret

class MergetSort:
    def mergeSort(self, nums):
        def _sort(nums, l, r):
            if l==r:
                return nums[l]

            mid = (l + r) // 2
            _sort(nums, l, mid)
            _sort(nums, mid+1, r)

            tmp = []
            i, j = l, mid+1
            while i<=mid or j<=r:
                if (i>mid or (j <= r and nums[j]<nums[i])):
                    tmp.append(nums[j])
                    j += 1
                else:
                    tmp.append(nums[i])
                    i += 1
            nums[l : r+1] = tmp
        _sort(nums, 0, len(nums)-1)
        return nums

class Rotate:
    # 轮转数组 in-place
    def rotate(self, nums, k):
        def swap(nums, l, r):
            while l<r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1

        k %= len(nums)

        swap(nums, 0, len(nums)-1)
        swap(nums, 0, k-1)
        swap(nums, k, len(nums)-1)

class SearchMatrix:
    def searchMatrix(self, matrix, target):
        m, n = len(matrix), len(matrix[0])
        i, j = 0, n-1
        while (i>=0 and i<m) and (j>=0 and j<n):
            if matrix[i][j] == target:
                return True

            if matrix[i][j] > target:
                j -= 1
            else:
                i += 1

        return False

class permuteUnique:
    def permuteUnique(self, nums):
        self.ans = []

        def dfs(nums, visited, path, depth, size):
            if depth == size:
                self.ans.append(path[:])
                return

            for i in range(size):
                if visited[i]: 
                    continue
                if i>0 and nums[i]==nums[i-1] and visited[i-1] == False:
                    continue
                visited[i] = True
                path.append(nums[i])
                dfs(nums, visited, path, depth+1, size)
                path.pop()
                visited[i] = False

        nums.sort()
        size = len(nums)
        visited = [ False ] * size
        path = []
        dfs(nums, visited, path, i, size)

        return self.ans

class FindRepeateNum:
    # 查找重复数
    def findRepeatenum(self, nums):
        temp = set()
        for num in nums:
            if num in temp:
                return num
            temp.add(num)
        return -1

    def findRepeatenum(self, nums):
        i, n = 0, len(nums)
        while i<n:
            if nums[i] == i:
                i += 1
                continue
            if nums[nums[i]] == nums[i]:
                return nums[i]
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]

        return -1

class Reverse:
    # 数字翻转 123 -> 321
    def reverse(self, num):
        INT_MIN, INT_MAX = -2**31, 2**31-1
        rev = 0
        while num != 0:
            # INT_MIN 也是一个负数，不能写成 rev < INT_MIN // 10
            if rev < INT_MIN // 10 + 1 or rev > INT_MAX // 10:
                return 0
            digit = num % 10
            # Python3 的取模运算在 x 为负数时也会返回 [0, 9) 以内的结果，因此这里需要进行特殊判断
            if num<0 and digit>0:
                digit -= 10

            # 同理，Python3 的整数除法在 x 为负数时会向下（更小的负数）取整，因此不能写成 x //= 10
            num = (num-digit) // 10
            rev = rev * 10 + digit
        return rev


class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.que = nums
        heapq.heapify(self.que)

    def add(self, val):
        heapq.heappush(self.q, val)
        while len(self.que) > self.k:
            heapq.heappop(self.que)

        rturn self.que[0]

class Compress:
    def compress(self, chars):
        n = len(chars)
        j, cnt = 0, 1
        for i in range(n):
            if i==n-1 or chars[i]!=chars[i+1]:
                chars[j] = chars[i]
                j += 1
                if cnt > 1:
                    for k in str(cnt):
                        chars[j] = k
                        j += 1
                cnt = 1
            else:
                cnt += 1
        return j

class Solution:
    # 至少有k个重复字符的最长子串
    def longestSubstring(self, s, k):
        def dfs(s, k):
            cnt = [0]*26
            for c in s:
                cnt[ord(c)-ord('a')] += 1
            split_char = ""
            for i in range(26):
                count = cnt[i]
                if count>0 and count<k:
                    split_char = chr(ord('a')+i)
                    break
            if split_char == "":
                return len(s)
            splits = s.split(split_char)
            _max = 0
            for split_s in splits:
                length = dfs(split_s, k)
                _max = max(_max, length)

            return _max
        
        return dfs(s, k)

def getIntersectionNode:
    # 两个链表第一个公共节点
    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return None
        p, q = headA, headB
        while(p!=q):
            p = p.next if p else headB
            q = q.next if q else headA

        return p

def OrangeRotting:
    def orangeRotting(self, grid):
        m, n = len(grid), len(grid[0])
        stack = []
        
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    count += 1
                elif grid[i][j] == 2:
                    stack.append( (i,j) )
        round = 0
        while count>0 and len(stack)>0:
            round += 1
            size = len(stack)
            for i in range(size):
                (i, j) = stack.pop(0)
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    _i, _j = i+dx, j+dy
                    if (_i>=0 and _i<m) and (_j>=0 and _j<n) and grid[_i][_j] == 1:
                        grid[_i][j] = 2
                        count -= 1
                        stack.append( (_i, _j) )
        return -1 if count>0 else round

class ReverseString:
    # In place 字符串反转
    def reverseString(self, s):
        l, r = 0, len(s)-1
        while l<r:
            s[l], s[r] = s[r], s[l]
            l += 1
            r -= 1

class ReplaceElements:
    # 替换为右侧最大值
    def replaceElements(self, arr):
        n, ans = -1, []
        for i in range(len(arr)-1, -1, -1):
            ans.append(n)
            if arr[i]>n:
                n = arr[i]
        return ans[::-1]

class Solution:
    # 段式回文串
    def longestDecomposition(self, text: str) -> int:
        if len(text) <= 1:
            return len(text)
        for i in range(1, len(text) // 2 + 1):
            if text[:i] == text[-i:]:
                return 2 + self.longestDecomposition(text[i: -i])
        return 1

class MiddleNode:
    # 链表中间节点
    def middleNode(self, head):
        if not head:
            return head
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow.next

class IssameTree:
    # 相同树
    def isSameTree(self, p, q):
        if not p and not q:
            return True
        if (not p or not q) or (p.val!=q.val):
            return False

        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

class SplitIntoFibonacci:
    # 数组拆分成斐波那契序列
    def splitIntoFibonaci(self, num):
        ans = []
        def backtrack(index):
            if index == len(num):
                return len(ans)>=3

            curr = 0
            for i in range(index, len(num)):
                if i>index and num[index] == "0":
                    break
                curr = curr*10 + ord(num[i])-ord('0')
                if curr > 2**31 -1:
                    break

                if len(ans)<2 or curr==ans[-2]+ans[-1]:
                    ans.append(curr)
                    if backtrack(i+1):
                        return True
                    ans.pop()
                elif len(ans)>2 and curr != ans[-2]+ans[-1]:
                    break
            return False
        
        backtrace(0)
        return ans

class VerifyPostorder:
    # 判断二叉搜索树的后序遍历
    def verifyPostorder(self, postorder):
        if not postorder:
            return False

        def recur(i, j):
            if i>=j:
                return True
            p = i
            while postorder[p] < postorder[j]:
                p += 1
            m = p
            while postorder[p] > postorder[j]:
                p += 1
            return p==j and recur(i, m-1) and recur(m, j-1)

        return recur(0, len(postorder)-1)

class Permutation:
    # 字符串排列组合
    def permutation(self, s):
        size = len(s)
        visited = [ False ] * len(s)
        s = sorted(s)
        self.ans = []
        def trace(s, depth, size, perm):
            if depth==size:
                self.ans.append(perm)
                return
            for j in range(size):
                if visited[j] or (j>0 and visited[j-1] and s[j-1]==s[j]):
                    continue
                visited[j] = True
                perm += s[j]
                trace(s, depth+1, size, perm)
                perm = perm[:-1]
                visited[j] = False
        perm = ''
        trace(s, 0, size, perm)
        return self.ans

class InsertIntoBST:
    # 插入二叉搜索树
    def insertIntoBST(self, root, val):
        if not root:
            return TreeNode(val)
        
        pos = root
        while pos:
            if val < pos.val:
                if not pos.left:
                    pos.left = TreeNode(val)
                    break
                else:
                    pos = pos.left
            else:
                if not pos.right:
                    pos.right = TreeNode(val)
                    break
                else:
                    pos = pos.right
        return root

class MaxRepOpt1:
    # 单字符重复子串最大长度
    def maxRepOpt1(self, text):
        cnt = Count(text)
        res = 0
        i = 0
        while i<len(text):
            j = i
            while j<len(text) and text[j]==text[i]:
                j += 1

            # cur_cnt = (j-i)
            # if cur_cnt < cnt[text[i]] and (j<len(text) or i>0):
            #     res = max(res, cur_cnt+1)

            k = j+1
            while k<len(text) and text[k]==text[i]:
                k += 1
            res = max(res, min(k-i, cnt[text[i]]))

        return res

class Solution:
    # 八皇后
    def solveNQueens(self, n: int) -> List[List[str]]:
        def generateBoard():
            board = list()
            for i in range(n):
                row[queens[i]] = "Q"
                board.append("".join(row))
                row[queens[i]] = "."
            return board

        def backtrack(row: int):
            if row == n:
                board = generateBoard()
                solutions.append(board)
            else:
                for i in range(n):
                    if i in columns or row - i in diagonal1 or row + i in diagonal2:
                        continue
                    queens[row] = i
                    columns.add(i)
                    diagonal1.add(row - i)
                    diagonal2.add(row + i)
                    backtrack(row + 1)
                    columns.remove(i)
                    diagonal1.remove(row - i)
                    diagonal2.remove(row + i)

        solutions = list()
        queens = [-1] * n
        columns = set()
        diagonal1 = set()
        diagonal2 = set()
        row = ["."] * n
        backtrack(0)
        return solutions

class Solution:
    # 解数独
    def solveSudoku(self, board: List[List[str]]) -> None:
        def dfs(pos: int):
            nonlocal valid
            if pos == len(spaces):
                valid = True
                return
            
            i, j = spaces[pos]
            for digit in range(9):
                if line[i][digit] == column[j][digit] == block[i // 3][j // 3][digit] == False:
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True
                    board[i][j] = str(digit + 1)
                    dfs(pos + 1)
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = False
                if valid:
                    return
            
        line = [[False] * 9 for _ in range(9)]
        column = [[False] * 9 for _ in range(9)]
        block = [[[False] * 9 for _a in range(3)] for _b in range(3)]
        valid = False
        spaces = list()

        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    spaces.append((i, j))
                else:
                    digit = int(board[i][j]) - 1
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True

        dfs(0)

class SplitArray:
    # 分割数组最大值
    def splitArray(self, nums, m):
        def check(x):
            total, cnt = 0, 1
            for num in nums:
                if total+num>x:
                    cnt += 1
                    total = num
                else:
                    total += num
            return cnt <= m

        left = max(nums)
        right= sum(nums)
        while left < right:
            mid = (left+right) // 2
            if check(mid):
                right = mid
            else:
                left = mid+1

        return left

class DeleteNode:
    # 删除节点
    def deleteNode(self, head, value):
        dummy_node = ListNode(-1)
        dummy_node.next = head
        if not head:
            return head

        pre, cur = dummy_node, head
        while cur:
            if cur.val != value:
                pre = cur
                cur = cur.next
            else:
                pre.next = cur.next
                cur = cur.next
        return dummy_node.next

class SortArray:
    def sortArray_quick(self, nums: List[int]) -> List[int]:
        def partion(nums, left, right):
            if left>=right:
                return
            pivot = random.randint(left, right)
            nums[pivot], nums[right] = nums[right], nums[pivot]
            i = left
            for j in range(left, right):
                if nums[j]<nums[right]:
                    nums[j], nums[i] = nums[i], nums[j]
                    i += 1
            nums[i], nums[right] = nums[right], nums[i]
            partion(nums, left, i-1)
            partion(nums, i+1, right)

        partion(nums, 0, len(nums)-1)
        return nums

    def sortArray_heap(self, nums):
        def buildheap(i, size, nums):
            if i>=size:
                return
            left, right = 2*i+1, 2*i+2
            largest = i
            if left < size and nums[left]>nums[largest]:
                largest=left
            if right< size and nums[right]>nums[largest]:
                largest=right
            if largest!=i:
                nums[i], nums[largest] = nums[largest], nums[i]
                buildheap(largest, size, nums)
        n = len(nums)
        for i in range(n//2, -1, -1):
            buildheap(i, n, nums)
        for i in range(n-1, -1, -1):
            nums[0], nums[i] = nums[i], nums[0]
            buildheap(0, i, nums)
        return nums

    def sortArray(self, nums):
        def mergesort(left, right, nums):
            if left==right:
                return
            mid = (left+right)//2
            mergesort(left, mid, nums)
            mergesort(mid+1, right,nums)
            tmp = []
            i, j = left, mid+1
            while (i<=mid) or (j<=right):
                if i>mid or(j<=right and nums[j]<nums[i]):
                    tmp.append(nums[j])
                    j += 1
                else:
                    tmp.append(nums[i])
                    i += 1
            nums[left:right+1] = tmp
        mergesort(0, len(nums)-1, nums)
        return nums

class BSTIterator:
    def __init__(self, root):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def next(self):
        cur = self.stack.pop()
        node = cur.right
        while node:
            self.stack.append(node)
            node = node.left
        return cur.val

    def hasNext(self):
        return len(self.stack) > 0

class Solution:
    # 最大的以 1 为边界的正方形
    def largestBorderedSquare(self, grid):
        m, n = len(grid), len(grid[0])
        left = [[0]*(n+1) for _ in range(m+1)] # 保存当前点左侧最长
        up = [[0]*(n+1) for _ in range(m+1)]   # 保存当前点上侧最长
        max_border = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if grid[i-1][j-1]:
                    left[i][j] = left[i][j-1]+1
                    up[i][j] = up[i-1][j] + 1
                    border = min(left[i][j], up[i][j])
                    while left[i-border+1][j] < border or up[i][j-border+1] < border:
                        border -= 1
                    max_border = max(max_border, border)
        return max_border ** 2

class FlipEquiv:
    # 翻转等价二叉树
    def flipEquiv(self, root1, root2):
        if root1 is root2:
            return True
        if not root1 or not root2 or root1.val!=root2.val:
            return False
        
        return (self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right)) or \
                (self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left))

class LevelOrderBottom:
    def levelOrderBottom(self, root):
        self.ans = []
        if not root:
            return []
        stack = [root]
        while stack:
            size = len(stack)
            tmp = []
            for i in range(size):
                node = stack.pop(0)
                tmp.append(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
            self.ans.append(tmp)
        return self.ans[::-1]


class MaxNum:
    # 拼接最大数: 从nums1 和 nums2 中分别选择k1, k2 个数(k1+k2=k)，使得最大，等效于
    # (removeKdigits) 任务;
    def maxNum(self, nums1, nums2, k):
        def pick_max(nums, k):
            stack = []
            drop = len(nums)-k
            for num in nums:
                while drop and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)
            return stack[:k]
        def merge(A, B):
            ans = []
            while A or B:
                bigger = A if A>B else B
                ans.append(bigger.pop(0))
            return ans

        return max( merge(pick_max(nums1, i), pick_max(nums2, k-i)) for i in range(k+1) if i<=len(nums1) and k-i<=len(nums2) )

class FindMaxLength:
    def findmaxLength(self, nums):
        counter= 0
        maxlen = 0
        prefix_counter = dict()
        prefix_counter[counter] = -1
        for i in range(len(nums)):
            if nums[i] == 0:
                counter -= 1
            else:
                counter += 1
            if counter not in prefix_counter:
                prefix_counter[counter] = i
            else:
                pi = prefix_counter[counter]
                maxlen = max(maxlen, i-pi)
        return maxlen

class IsInterleave:
    # 交错字符串-> 动态规划
    def isInterleave(self, s1, s2, s3):
        m, n, k = len(s1), len(s2), len(s3)
        dp = [ [False]*(n+1) for _ in range(m+1) ]

        if m+n != k:
            return False
        dp[0][0] = True
        for i in range(m+1):
            for j in range(n+1):
                p = i+j-1
                if i>0:
                    dp[i][j] = (dp[i-1][j] and s1[i-1]==s3[p])
                if j>0:
                    dp[i][j] |= (dp[i][j-1] and s2[j-1]==s3[p])
        return dp[m][n]

class CountPrimes:
    # 计算质数
    def countPrimes(self, n):
        def check(x):
            i=2
            while i*i <= x:
                if x % i == 0:
                    return False
            return True

        ans = 0
        for i in range(2, n):
            if check(i):
                ans += 1
        return ans

class IsHappy:
    # 判断快乐数
    def get_next(self, n):
        total_sum = 0
        while n>0:
            n, digit = divmod(n, 10)
            total_sum += digit**2
        return total_sum

    def isHappy(self, n):
        seen = set()
        # 判断是否循环出现
        while n!=1 and n not in seen:
            seen.add(n)
            n = self.get_next(n)
        return n==1

    def isHappy_v2(self, n):
        # 循环判断可以通过快慢指针来处理
        slow_runner = n
        fast_runner = self.get_next(n)
        while fast_runner!=1 and fast_runner!=slow_runner:
            slow_runner = self.get_next(slow_runner)
            fast_runner = self.get_next( self.get_next(fast_runner) )
        return fast_runner==1

class MergeInBetween:
    # 链表合并
    def mergeInBetween(self, list1, a, b, list2):
        dummynode = ListNode(-1)
        dummynode.next = list1
        pre, cur = dummynode, list1
        for i in range(a):
            pre = cur
            cur = cur.next
        phead = cur
        for i in range(a, b+1):
            phead = phead.next
        pre.next = list2
        cur = list2
        while cur.next:
            cur = cur.next
        cur.next = phead

        return dummynode.next
    def mergeInBetween(self, list1, a, b, list2):
        left_pointer, right_pointer = list1, list1
        for i in range(a-1):
            left_pointer = left_pointer.next
        for i in range(b+1):
            right_pointer= right_pointer.next

        left_pointer.next = list2
        last_pointer = list2
        while last_pointer.next is not None:
            last_pointer = last_pointer.next
        last_pointer.next = right_pointer
        return list1

class ConstructFromPrePost:
    # 根据前向和后向结果生成二叉树
    def constructFromPrePost(self, pre, post):
        if not pre:
            return None
        root = TreeNode(preorder[0])
        if len(pre)==1:
            return root
        L = post.index(pre[1]) + 1
        root.left = self.constructFromPrePost(pre[1:L+1], post[:L])
        root.right= self.constructFromPrePost(pre[L+1:], post[L:-1])
        return root

class MaxIncreaseKeepingSkyline:
    # 保持天际线
    def maxIncreaseKeepingSkyline(self, grid):
        m, n = len(grid), len(grid)[0]
        ans = 0
        for i in range(m):
            for j in range(n):
                row = grid[i][:]
                col = [grid[_][j] for _ in range(n)]
                ans += (min(max(row), max(col))-grid[i][j])
        return ans

class letterCombinations:
    # 电话号码字母组合数
    phoneMap = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }
    def letterCombinations(self, digits):
        if not digits:
            return []
        combine, combines = [], []
        def backtrace(inx):
            if inx==len(digits):
                combines.append(''.join(combine))
            else:
                for letter in phoneMap[digits[inx]]:
                    combine.append(letter)
                    backtrack(inx+1)
                    combine.pop()
        backtrack(0)
        return combines

class RepeateSubstringPattern:
    # 重复字符串
    def repeateSubstringPattern(self, s):
        return (s+s).find(s, 1) == len(s)

    def repeateSubstringPattern(self, s):
        n = len(s)
        for i in range(1, n//2+1):
            if n%i==0:
                if all(s[j]==s[j-i] for j in range(i,n)):
                    return True
        return False

class SingleNumber:
    # 重复数字
    def singleNumber(self, nums):
        freq = Counter(nums)
        return [num for num,cnt in freq.items() if cnt==1]

    def singleNumber(self, nums):
        x_or_sum = 0
        for num in nums:
            x_or_sum ^= num

        low_sb = x_or_sum & (-x_or_sum)   # 返回最低位为1 参考: https://blog.csdn.net/oyoung_2012/article/details/79932394

        type1 = type2 = 0
        for num in nums:
            if num & low_sb:
                type1 ^= num
            else:
                type2 ^= num
        return [type1, type2]

from sortedcontainers import SortedList
class LongestSubarray:
    # 绝对差不超过限制的最长连续子数组
    def longestSubarray(self, nums, limit):
        s = SortedList()
        n = len(nums)
        left = right = ret = 0

        while right<n:
            s.add(nums[right])
            while s[-1]-s[0] >limit:
                s.remove(nums[left])
                left += 1
            ret = max(ret, right-left+1)
            right += 1

        return ret

class IceBreakingGame:
    # 圆圈中剩下的最后数字 / 破冰游戏，【逆向思维，从最后一个往前推理, pos指的是下标】
    def iceBreakingGame(self, nums, target):
        n = len(nums)
        pos = 0
        for i in range(2, n+1):
            pos = (pos + target) % i
        return pos

class IsRectangleOverlap:
    # 矩形重叠
    def isRectangleOverlap(self, rec1, rec2):
        def check(p_left, p_right, q_left, q_right):
            return min(p_right, q_right) > max(p_left, q_left)

        return check(rec1[0], rec1[2], rec2[0], rec2[2]) and \
                check(rec1[1], rec1[3], rec2[1], rec2[3])

class DeepestLeavesSum:
    def deepestLeavesSum(self, root):
        if not root:
            return 0
        stack = [root]
        while stack:
            tmp = []
            size = len(stack)
            for i in range(size):
                node = stack.pop(0)
                tmp.append(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

            if not stack:
                return sum(tmp)

class FindDisappearedNumbers:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        return list(set([i+1 for i in range(len(nums))]) - set(nums))

class LargestNumber:
    def largestNumber(self, nums):
        def _sort(x, y):
            x, y = str(x), str(y)
            if x+y > y+x:
                return 1
            elif x+y < y+x:
                return -1
            else:
                return 0
        nums = sorted(nums, key=cmp_to_key(_sort), reverse=True)
        if str(nums[0]) == "0":
            return "0"
        return ''.join([str(n) for n in nums])

class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [ [0]*(n) for _ in range(m) ]
        dp[0][0] = 0 if obstacleGrid[0][0]==1 else 1
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] if obstacleGrid[i][0]!=1 else 0
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] if obstacleGrid[0][j]!=1 else 0

        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j]==1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]

class Solution:
    # 整数替换
    def integerReplacement(self, n: int) -> int:
        @cache
        def dfs(n):
            if n==1:
                return 0
            if n%2==0:
                return dfs(n//2) + 1
            else:
                return min(dfs(n//2), dfs(n//2+1)) + 1
        return dfs(n)

class Solution:
    # 寻找最近回文数
    def nearestPalindromic(self, n: str) -> str:
        m = len(n)
        candidates = [10 ** (m - 1) - 1, 10 ** m + 1]
        selfPrefix = int(n[:(m + 1) // 2])
        for x in range(selfPrefix - 1, selfPrefix + 2):
            y = x if m % 2 == 0 else x // 10
            while y:
                x = x * 10 + y % 10
                y //= 10
            candidates.append(x)

        ans = -1
        selfNumber = int(n)
        for candidate in candidates:
            if candidate != selfNumber:
                if ans == -1 or \
                        abs(candidate - selfNumber) < abs(ans - selfNumber) or \
                        abs(candidate - selfNumber) == abs(ans - selfNumber) and candidate < ans:
                    ans = candidate
        return str(ans)

class RemoveDuplicates:
    # 删除有序数组重复数字(最多保留一个)
    def removeDuplicates(self, nums):
        if not nums:
            return 0
        p, q = 1, 1
        while q<len(nums):
            if (nums[p-1]!=nums[q]):
                nums[p] = nums[q]
                p += 1
            q += 1
        return p

    # 删除有序数组重复数字(最多保留两个)
    def removeDuplicates(self, nums):
        size = len(nums)
        if size<=2:
            return size
        slow, fast = 2, 2
        while fast<size:
            if nums[slow-2] != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow

class MinOperations:
    def minOperations(self, s1, s2, x):
        n = len(s1)
        idxs = [i for i in range(n) if s1[i]!=s2[i]]
        k = len(idxs)
        if k % 2:
            return -1
        dp = [inf] * (k+1)
        dp[0] = 0
        for i in range(k):
            if i%2==0:
                dp[i+1] = dp[i]
            else:
                dp[i+1] = dp[i] + x
            if i:
                dp[i+1] = min(dp[i+1], dp[i-1]+idxs[i]-idxs[i-1])
        return dp[k]

    def minOperations(self, s1, s2, x):
        if s1.count('1') % 2 != s2.count('1') %2:
            return -1

        @cache
        def dfs(i, j, pre_rev):
            if i<0:
                return inf if j or per_rev else 0
            if (s1[i] == s2[i]) == (not pre_rev):
                return dfs(i-1, j, False)
            res = min(dfs(i-1, j+1, False)+x, dfs(i-1, j, True)+1)
            if j:
                res = min(res, dfs(i-1, j-1, False))
            return res

        return dfs(len(s1)-1, 0, False)

class Combine:
    # 组合
    def combine(self, n, k):
        ans = []
        def dfs(i, path, depth):
            if depth == k:
                ans.append(path[:])
                return

            for d in range(i, n+1):
                if d in set(path):
                    continue
                path.append(d)
                dfs(d+1, path, depth+1)
                path.pop()
        path = []
        dfs(1, path, 0)
        return ans

    def combine(self, n, k):
        ans, path = [], []
        def dfs(start_index, path):
            if len(path) == k:
                ans.append(path[:])
                return
            size = n - (k-len(path)) + 1    # 减少回溯
            for i in range(start_index, size+1):
                path.append(i)
                dfs(i+1, path)
                path.pop()
        dfs(1, path)
        return ans

class Solution:
    # 打乱数组, 保证概率相同
    def __init__(self, nums):
        self.nums = nums
        self.original = nums.copy()

    def reset(self):
        self.nums = self.original.copy()
        return self.nums

    def shuffle(self):
        for i in range(len(self.nums)):
            j = random.randrange(i, len(self.nums))
            self.nums[i], self.nums[j] = self.nums[j], self.nums[i]
        return self.nums

class CanFinish:
    # 课程表问题 判断 DAG
    def canFinish(self, numCourse, prerequisites):
        indegrees = [0 for _ in range(numCourse)]
        adjacency = [[] for _ in range(numCourse)]
        queue = deque()

        for cur, pre in prerequisites:
            indegrees[cur] += 1
            adjacency[pre].append(cur)

        for i in range(len(indegrees)):
            if not indegrees[i]:
                queue.append(i)

        while queue:
            pre = queue.popleft()
            numCourse -= 1
            for cur in adjacency[pre]:
                indegrees[cur] -= 1
                if not indegrees[cur]: queue.append(cur)

        return not numCourse

    def canFinish(self, numCourses, prerequisites):
        def dfs(i, adjacency, flags):
            if flags[i] == -1:
                return True
            if flags[i] == 1:
                return False

            flags[i] = 1
            for j in adjacency[i]:
                if not dfs(j, adjacency, flags): return False
            flags[i] = -1
            return True

        adjacency = [ [] for _ in range(numCourses) ]
        flags = [0 for _ in range(numCourses)]
        for cur, pre in prerequisites:
            adjacency[pre].append(cur)
        for i in range(numCourses):
            if not dfs(i, adjacency, flags): return False
        return True

class checkSymmetricTree:
    def checkSymmetricTree(self, root):
        def check(left, right):
            if not left and not right:
                return True
            if not left or not right or left.val != right.val:
                return False
            return check(left.left, right.right) and check(left.right, right.left)
        if not root:
            return True
        return check(root.left, root.right)
            
class LowestCommonAncestor:
    def lowestCommonAncestor(self, root, p, q):
        if not root or root.val==p.val or root.val==q.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right= self.lowestCommonAncestor(root.right, p, q)
        if not left:   # 说明两个节点都在right
            return right
        if not right:
            return left
        return root    # 说明两个节点分别在左右两边

class shiftingLetters:
    def shiftingLetters(self, s, shifts):
        ans = []
        prefixsum = sum(shifts) % 26
        for i, c in enumerate(s):
            ans.append(chr(ord('a') + (ord(c)-ord('a') + prefixsum) % 26))
            prefixsum = (prefixsum - shifts[i]) % 26

        return "".join(ans)

class Convert:
    # N 字形变换
    def convert(self, s, numRows):     # 模拟二维矩阵位置
        maxcol = len(s)//numRows + 1
        if numRows == 1 or numRows>=len(s):
            return s
        t = numRows * 2 - 2
        col = (len(s)+t-1) // t * (numRows-1)
        grid = [ ['']*c for _ in range(numRows) ]

        m, n = 0, 0
        for c in s:
            grid[m][n] = c
            if n%(numRows-1) == 0:
                if m<(numRows-1):
                    m += 1
                else:
                    m -= 1
                    n += 1
            else:
                n += 1
                m -= 1

        return ''.join(ch for row in grid for ch in row if ch)

    def convert(self, s, numRows):
        r = numRows
        if r==1 or r>=len(s):
            return s
        max = [ [] for _ in range(r) ]
        t, x = r*2-2, 0
        for i, ch in enumerate(s):
            max[x].append(ch)
            x += 1 if i%t<r-1 else -1
        return ''.join(ch for row in grid for ch in row if ch)

class CheckInclusion:
    def checkInclusion(self, s1, s2):
        if len(s1)>len(s2):
            return False
        counter1 = collections.Counter(s1)
        n = len(s2)
        left, right = 0, len(s1)-1
        counter2 = collections.Counter(s2[0:right])
        while right<n:
            counter2[s2[right]] += 1
            if counter2==counter1:
                return True
            counter2[s2[left]] -= 1
            if counter2[s2[left]] == 0:
                del counter2[s2[left]]
            left += 1
            right+= 1

        return False

class nextGreaterElement:
    def nextGreaterElement(self, nums1, nums2):
        next_g, stack = dict(), []
        for num in reversed(nums2):
            while stack and num > stack[-1]:
                stack.pop()
            next_g[num] = stack[-1] if stack else -1
            stack.append(num)
        return [next_g[num] for num in nums1]

class NumsSameConsecDiff:
    # 连续差相同的数字
    def numsSameConsecDiff(self, n, k):
        q = deque(range(1, 10))

        while n>1:
            size = len(q)
            for _ in range(size):
                u = q.popleft()
                for v in {u % 10 - k, u % 10 + k}:  # 不用集合就判断k=0
                    if 0<=v<=9:
                        q.append(u*10 + v)
            n -= 1

        return list(q)

class LongestDiverseString:
    # 最长快乐数
    def longestDiverseString(self, a, b, c):
        ans = []
        c_cnt = [[a, 'a'], [b, 'b'], [c, 'c']]
        while True:
            c_cnt.sort(key= lambda x: -x[0])
            hasNext = False
            for i, (cnt, ch) in enumerate(c_cnt):
                if cnt<=0:
                    break
                if len(ans)>=2 and ans[-2]==ch and ans[-1]==ch:
                    continue
                hasNext = True
                ans.append(ch)
                c_cnt[i][0] -= 1
                break

            if not hasNext:
                return ''.join(ans)

class FindLengthOfShortestSubarray:
    # 删除最短子数组使有序
    def findLengthOfShortestSubarray(self, arr):
        n = len(arr)
        right = n-1
        while right and arr[right-1]<=arr[right]:
            right -= 1
        if right==0:
            return 0
        # 此时有arr[right-1] > arr[right]
        ans = right
        left= 0
        while left==0 or arr[left-1] <= arr[left]:
            while right<n and arr[left] > arr[right]:
                right += 1
            # 此时 arr[left] <= arr[right], 删除[left+1:right]
            ans = min(ans, right-left-1)
            left += 1
        return ans

class LargestTimeFromDigits:
    def largestTimeFromDigits(self, arr):
        ans = -1
        max_hours, max_mins = -1, -1
        for h1, h2, m1, m2 in itertools.permutations(arr):
            hours = 10*h1 + h2
            mins = 10*m1 + m2
            time = 60 * hours + mins
            if 0<=hours<24 and 0<=mins<60 and time>ans:
                ans = time
                max_hours, max_mins = hours, mins
        return f"{max_hours}:{max_mins}" if ans>=0 else ""

class Rob:
    # 打家劫舍
    def rob(self, nums):
        n = len(nums)
        if n==1:
            return nums[0]
        if n==2:
            return max(nums)

        dp = [ 0 ]*n
        dp[0], dp[1] = nums[0], max(dp[0], dp[1])
        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        return dp[-1]

    # 进阶版(环形), 解决思路：将环形看做两个非环形组合，(选取第一个和选取最后一个)
    # 然后分别处理，选择最大值即可
    def rob(self, nums):
        def _rob(nums):
            cur, pre = 0, 0
            for num in nums:
                cur, pre = max(cur, pre+num), cur
            return cur
        
        return max(_rob(nums[:-1], _rob(nums[1:]))) if len(nums)!=1 else nums[0]

class InvertoryManagement:
    def invertoryManagement(self, stock, cnt):
        stock = sorted(stock)
        return stock[:cnt]

    def invertoryManagement(self, stock, cnt):
        if cnt==0:
            return []
        hp = [-x for x in stock[:cnt]]
        heapq.heapify(hp)
        for i in range(cnt, len(stock)):
            if -hp[0] > stock[i]:
                heapq.heappop(hp)
                heqpq.heappush(hp, -stock[i])
        ans = [-x for x in hp]
        return ans

class FinalPrice:
    def finalPrice(self, prices):
        # [8, 4, 6, 2, 3]
        ans = [-1]*len(prices)
        stack = []
        for i in range(len(prices)-1, -1, -1):
            while stack and stack[-1] > prices[i]:
                stack.pop()
            if stack:
                ans[i] = stack[-1]
            stack.append(prices[i])
        _prices = []
        for i,price in enumerate(prices):
            _price = price if ans[i]==-1 else price-ans[i]
            _prices.append(_price)
        return _prices

class KMP:
    def build_next(self, pattern):
        n = len(pattern)
        next_table = [0]
        j, i = 0, 1
        while i<n:
            if pattern[i] == pattern[j]:
                j += 1
                i += 1
                next_table.append(j)
            else:
                if j==0:
                    next_table.append(0)
                else:
                    j = next_table[j-1]
        return next_table

    def search_location(self, source, pattern):
        next_table = self.build_next(pattern)
        i, j = 0, 0
        locations = []
        while i<len(source):
            if source[i]==pattern[j]:
                i += 1
                j += 1
                if j==len(pattern):
                    locations.append( i-j )
                    j = next_table[-1]

            elif j==0:
                i += 1

            else:
                j = next_table[j-1]

        return locations

class WordBreak:
    # 单词拆分
    def wordBreak(self, s, wordDict):
        
        @lru_cache(None)
        def backtrack(index):
            if index==len(s):
                return [[]]
            ans = []
            for i in range(index+1, len(s)+1):
                word = s[index:i]
                if word in wordSet:
                    nextWordBreaks = backtrack(i)
                    for nextWordBreak in nextWordBreaks:
                        ans.append(nextWordBreak.copy() + [word])
            return ans

        @lru_cache(None)
        def backtrackv2(index, path):   # unhashable type: list
            path = list(path)
            if index==len(s):
                self.ans.append(path[:])
                return

            for i in range(index+1, len(s)+1):
                word = s[index:i]
                if word in wordSet:
                    path.append(word)
                    backtrack(i, tuple(path))
                    path.pop()
        
        wordSet = set(wordDict)
        breakList = backtrack(0)
        return [" ".join(words[::-1]) for words in breakList]

    def wordBreak_v2(self, s, wordDict):
        kmp = KMP()
        local_index = [ [] for _ in range(len(s)) ]
        for i, word in enumerate(wordDict):
            locations = kmp.search_location(s, word)
            for location in locations:
                local_index[location].append(i)

        result, r = [], []
        def dfs(index):
            if index == len(s):
                _s = ' '.join(r)
                result.append(_s)
            for j in local_index[index]:
                r.append(wordDict[j])
                dfs(index+len(wordDict[j]))
                r.pop()
        dfs(0)
        return result

class NestedIterator:
    def __init__(self, nestedList):
        self.queue = collections.deque()
        def helper(nests):
            for nest in nests:
                if nest.isInteger():
                    self.queue.append(nest.getInteger())
                else:
                    helper(nest.getList())
        helper(nestedList)

    def next(self):
        return self.queue.popleft()

    def hasNext(self):
        return len(self.queue) > 0

class NestedIterator:
    def __init__(self, nestedList):
        self.stack = []
        for i in range(len(nestedList)-1, -1, -1):
            self.stack.append(nestedList[i])

    def next(self):
        cur = self.stack.pop()
        return cur.getInteger()

    def hasNext(self):
        while self.stack:
            if self.stack[-1].isInteger():
                return True
            nest = self.stack.pop()
            for i in range(len(nest.getList())-1, -1, -1):
                self.stack.append(nest.getList()[i])

        return False

class ShortestPalindrome:
    # 最短回文串
    def shortestPalidrome(self, s):
        n = len(s)
        fail = [-1] * n
        for i in range(1, n):
            j = fail[i-1]
            while j != -1 and s[j+1] != s[i]:
                j = fail[j]
            if s[j+1] == s[i]:
                fail[i] = j + 1

        best = -1
        for i in range(n-1, -1, -1):
            while best != -1 and s[best+1] !=s[i]:
                best = fail[best]
            if s[best+1] == s[i]:
                best += 1

        add = ("" if best==n-1 else s[best+1:])
        return add[::-1] + s

    def shortestPalidrome_v2(self, s):
        n = len(s)
        # build next table
        fail = [ 0 ] * n
        for i in range(1,n):
            j = fail[i-1]
            while j>0 and s[j] != s[i]:
                j = fail[j-1]
            if s[j] == s[i]:
                fail[i] = j+1

        # 进行匹配，获取匹配位置
        s1 = s[::-1]
        j = 0
        for i in range(n):
            while j>0 and s1[i]!=s[j]:
                j = fail[j-1]

            if s1[i] == s[j]:
                j += 1

        add = s[j:]
        return add[::-1] + s

class ValidPalindrome:
    # 是否回文串，(可删除一个)
    def validPalindrome(self, s):
        n = len(s)
        def check(left, right):
            while left<=right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True

        left, right = 0, n-1
        while left <= right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return check(left, right-1) or check(left+1, right)

        return True

class Solution:
    # N 叉树遍历
    def preorder(self, root):
        if not root:
            return []
        self.ans = []
        def dfs(root):
            if not root:
                return
            self.ans.append(root.val)
            for node in root.children:
                dfs(node)
        return self.ans

    def preorder(self, root):
        if not root:
            return []
        self.ans = []
        stack = [root]
        while stack:
            root = stack.pop()
            self.ans.append(root.val)
            for node in root.children[::-1]:
                stack.append(node)

        return self.ans

class FindMaxConsecutiveOnes:
    def findMaxConsecutiveOnes(self, nums):
        count, maxcount = 0, 0
        for num in nums:
            if num==1:
                count += 1
            else:
                maxcount = max(maxcount, count)
                count = 0
        maxcount = max(maxcount, count)
        return maxcount

class PathTarget:
    def pathTarget(self, root, target):
        if not root:
            return []
        self.ans = []
        def check(root, path, s):
            if not root:
                return 

            s += root.val
            path.append( root.val )
            if not root.left and not root.right:
                if s==target:
                    self.ans.append(path[:])
            check(root.left, path, s)
            check(root.right, path, s)
            s -= root.val
            path.pop()

        check(root, [], 0)

        return self.ans

class NumTrees:
    # 不同的二叉搜索树
    def numTrees(self, n):
        dp = [ 0 ] * (n+1)   # dp[i] 表示序列长度为i时对应的二叉搜索树个数
        dp[0], dp[1] = 1, 1
        for i in range(2, n+1):
            for j in range(1, i+1):
                dp[i] += dp[j-1] * dp[i-j]
        return dp[-1]

class RomanToInt:
    # 罗马 to int
    SYMBOL_VALUES = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
    }
    def romanToint(self, s):
        ans = 0
        for i in range(len(s)):
            value = self.SYMBOL_VALUES[s[i]]
            if i<len(s)-1 and value < self.SYMBOL_VALUES[s[i+1]]:
                ans -= value
            else:
                ans += value
        return ans

class ReverseParentheses:
    def reverseParentheses(self, s):
        stack = []
        for ch in s:
            if ch == ")":
                tmp = ''
                while stack[-1] != '(':
                    tmp += stack.pop()
                stack.pop()
                tmp = tmp[::-1]
                stack += list(tmp)
            else:
                stack.append(ch)
        return ''.join(stack[::-1])

class ThreeSumCloset:
    def threeSumCloset(self, nums, target):
        nums.sort()
        n = len(nums)
        nearest = float('inf')
        for i in range(n-2):
            if i>0 and nums[i]==num[i-1]:
                continue
            left, right = i+1, n-1
            while left < right:
                s = nums[left] + nums[right] + nums[i]
                if s==target:
                    return s
                if abs(s-target) < abs(nearest-target):
                    nearest = s
                if s>target:
                    right -= 1
                    while left<right and nums[right]==nums[right+1]:
                        right -= 1
                else:
                    left += 1
                    while left<right and nums[left]==nums[left-1]:
                        left += 1
        return nearest

class CombinationSum3:
    # 组合总和III
    def combinationSum3(self, k, n):
        self.ans = []
        def backtrack(inx, depth, path, target):
            if depth==k:
                if target==n:
                    self.ans.append(path[:])
                return

            for i in range(inx, 10):

                if (target + i) > n:
                    break

                path.append(i)
                backtrack(i+1, depth+1, path, target+i)
                path.pop()

        path = []
        backtrace(1, 0, path, 0)
        return self.ans

class Solution:
    def decorateRecord(self, root):
        if not root:
            return []
        self.ans = []
        stack = [ root ]
        while stack:
            size = len(stack)
            tmp = []
            for i in range(size):
                node = stack.pop(0)
                tmp.append(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
            self.ans.append( tmp )
        return self.ans

    def decorateRecord(self, root):
        self.depth_nodes = defaultdict(list)
        def dfs(depth, root):
            if not root:
                return
            self.depth_nodes[depth].append(root.val)
            dfs(depth+1, root.left)
            dfs(depth+1, root.right)
        dfs(0, root)
        
        depth_nodes_sorted = sorted(self.depth_nodes.items(), key=lambda x:x[0])
        self.ans = []
        for depth, nodes in depth_nodes_sorted:
            self.ans.append(nodes)
        return self.ans

from enum import Enum, auto

class ExprStatus(Enum):
    VALUE = auto()  # 初始状态
    NONE  = auto()  # 表达式类型未知
    LET   = auto()  # let 表达式
    LET1  = auto()  # let 表达式已经解析了 vi 变量
    LET2  = auto()  # let 表达式已经解析了最后一个表达式 expr
    ADD   = auto()  # add 表达式
    ADD1  = auto()  # add 表达式已经解析了 e1 表达式
    ADD2  = auto()  # add 表达式已经解析了 e2 表达式
    MULT  = auto()  # mult 表达式
    MULT1 = auto()  # mult 表达式已经解析了 e1 表达式
    MULT2 = auto()  # mult 表达式已经解析了 e2 表达式
    DONE  = auto()  # 解析完成

class Expr:
    __slots__ = 'status', 'var', 'value', 'e1', 'e2'

    def __init__(self, status):
        self.status = status
        self.var = ''  # let 的变量 vi
        self.value = 0  # VALUE 状态的数值，或者 LET2 状态最后一个表达式的数值
        self.e1 = self.e2 = 0  # add 或 mult 表达式的两个表达式 e1 和 e2 的数值

class Solution:
    def evaluate(self, expression: str) -> int:
        scope = defaultdict(list)

        def calculateToken(token: str) -> int:
            return scope[token][-1] if token[0].islower() else int(token)

        vars = []
        s = []
        cur = Expr(ExprStatus.VALUE)
        i, n = 0, len(expression)
        while i < n:
            if expression[i] == ' ':
                i += 1  # 去掉空格
                continue
            if expression[i] == '(':
                i += 1  # 去掉左括号
                s.append(cur)
                cur = Expr(ExprStatus.NONE)
                continue
            if expression[i] == ')':  # 本质上是把表达式转成一个 token
                i += 1  # 去掉右括号
                if cur.status is ExprStatus.LET2:
                    token = str(cur.value)
                    for var in vars[-1]:
                        scope[var].pop()  # 清除作用域
                    vars.pop()
                elif cur.status is ExprStatus.ADD2:
                    token = str(cur.e1 + cur.e2)
                else:
                    token = str(cur.e1 * cur.e2)
                cur = s.pop()  # 获取上层状态
            else:
                i0 = i
                while i < n and expression[i] != ' ' and expression[i] != ')':
                    i += 1
                token = expression[i0:i]

            if cur.status is ExprStatus.VALUE:
                cur.value = int(token)
                cur.status = ExprStatus.DONE
            elif cur.status is ExprStatus.NONE:
                if token == "let":
                    cur.status = ExprStatus.LET
                    vars.append([])  # 记录该层作用域的所有变量, 方便后续的清除
                elif token == "add":
                    cur.status = ExprStatus.ADD
                elif token == "mult":
                    cur.status = ExprStatus.MULT
            elif cur.status is ExprStatus.LET:
                if expression[i] == ')':  # let 表达式的最后一个 expr 表达式
                    cur.value = calculateToken(token)
                    cur.status = ExprStatus.LET2
                else:
                    cur.var = token
                    vars[-1].append(token)  # 记录该层作用域的所有变量, 方便后续的清除
                    cur.status = ExprStatus.LET1
            elif cur.status is ExprStatus.LET1:
                scope[cur.var].append(calculateToken(token))
                cur.status = ExprStatus.LET
            elif cur.status is ExprStatus.ADD:
                cur.e1 = calculateToken(token)
                cur.status = ExprStatus.ADD1
            elif cur.status is ExprStatus.ADD1:
                cur.e2 = calculateToken(token)
                cur.status = ExprStatus.ADD2
            elif cur.status is ExprStatus.MULT:
                cur.e1 = calculateToken(token)
                cur.status = ExprStatus.MULT1
            elif cur.status is ExprStatus.MULT1:
                cur.e2 = calculateToken(token)
                cur.status = ExprStatus.MULT2
        return cur.value

class SingleNumber:
    def singleNumber(self, nums):
        ans = 0
        for i in range(32):
            total = sum(num>>i for num in nums)
            if total % 3 :
                if i==31:
                    ans -= (1<<i)
                else:
                    ans |= (1<<i)

        return ans

    def singleNumber(self, nums: List[int]) -> int:
        low=0
        high=0
        for n in nums:
            # 计数+1
            carry=low&n
            low^=n
            high|=carry
            # 如果计数等于 3，重置为 0
            reset=low^high
            low&=reset
            high&=reset
        return low
    def singleNumber(self, nums):
        seen_once, seen_twice = 0, 0
        for num in nums:
            seen_once = ~seen_twice & (seen_once ^ num)
            seen_twice= ~seen_once & (seen_twice ^ num)

        return seen_once

class NthUglyNumber:
    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        i, j, k = 1, 1, 1
        ans, cnt = None, 0
        while cnt<n:
            _min = min(a*i, b*j, c*k)
            if _min == a*i:
                i+=1
            elif _min== b*j:
                j+=1
            else:
                k += 1
            if ans !=_min:
                cnt += 1
                ans = _min
        return ans

class Solution:
    def findLength(self, nums1, nums2):
        m, n = len(nums1), len(nums2)
        dp = [ [0]*(n+1) for _ in ragne(m+1) ]

        ans = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if nums1[i-1]==nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    ans = max(dp[i][j], ans)
                else:
                    dp[i][j] = 0
        return ans

class LargestMultipleOfThree:
    # 能被3整除最大数
    def largestMultipleOfThree(self, digits):
        dct = [ 0 ] * 10
        mct = [ 0 ] * 3
        mv = defaultdict(list)

        _sum = 0
        for d in digits:
            dct[d] += 1
            mct[d%3] += 1
            mv[d%3].append(d)
            _sum += d

        ans = ''
        if _sum % 3 == 0:  # 所有数字和刚好能被3整除
            for i in range(10):
                ans += str(i)*dct[i]
        elif _sum % 3 == 1:
            if mct[1] >= 1:
                d = sorted(mv[1])[0]  # 选择余数为1 的最小数
                dct[d] -= 1
                for i in range(10):
                    ans += str(i)*dct[i]
            elif mct[2] >= 2:
                ds = sorted(mv[2])[:2]
                for d in ds:
                    dct[d] -= 1
                for i in range(10):
                    ans += str(i)*dct[i]
        else:
            if mct[2] >= 1:
                d = sorted(mv[2])[0]
                dct[d] -= 1
                for i in range(10):
                    ans += str(i)*dct[i]
            elif mct[1] >= 2:
                ds = sorted(mv[1])[2]
                for d in ds:
                    dct[d] -= 1
                for i in range(10):
                    ans += str(i)*dct[i]
        temp = ans[::-1]
        if temp and temp[0]=="0":
            return "0"
        return temp

class MaximumSwap(object):
    def maximumswap(self, nums):
        num_str = list(str(num))
        length = len(num_str)
        for i in range(length):
            max_d, inx = num_str[i], i
            for j in range(length-1, i, -1):
                if num_str[j] > max_d:
                    max_d, inx = num_str[j], j
            if inx != i:
                num_str[i], num_str[inx] = num_str[inx], num_str[i]
                break
        return int(''.join(num_str))

class OnlineSoftmax:
    def online_softmax(self, nums):
        pre_max = -(float('inf'))
        _sum = 0.0
        for i in range(len(nums)):
            new_max = max(pre_max, nums[i])
            _sum  = _sum * math.exp(pre_max-new_max) + math.exp(nums[i]-new_max)
            pre_max = new_max

        dst = []
        for i in range(len(nums)):
            dst.append(math.exp( nums[i]-pre_max) / _sum )
        return dst

# 类消消乐字符串消除, 删除3个及以上相同字符
# 'aabbbac -> c,  abbbaac -> c'
class CharElimination:
    def charelimination(self, s):
        n = len(s)
        if n<=2:
            return s
        stack = [(s[0], 1)]
        i = 1
        while i<n:
            c = s[i]
            if not stack:
                stack.append((c, 1))
                i += 1
                continue

            if c!=stack[-1][0] and stack[-1][-1]>=3:
                for j in range(stack[-1][-1]):
                    stack.pop()

            if c == stack[-1][0]:
                count = stack[-1][-1] + 1
                stack.append((c, count))
                i += 1
            else:
                stack.append((c, 1))
                i += 1

        return ''.join([_[0] for _ in stack])

# 验证前序遍历序列二叉搜索树
class VerifyPreorder:
    def verifyPreorder(self, preorder):
        if len(preorder)<=2:
            return True
        stack = []
        _min = -float('inf')
        for i in range(len(preorder)):
            if preorder[i] < _min:
                return False
            while (stack and stack[-1] < preorder[i]) {
                _min = stack.pop()
            }

            stack.append(preorder[i])
        return True

# 二叉树层序遍历
class Solution:
    def level_travel1(self, root): # 遍历方式
        if not root:
            return []
        stack = [root]
        ans = []
        while stack:
            size = len(stack)
            tmp = []
            for i in range(size):
                node = stack.pop(0)
                tmp.append(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
                ans.append(tmp)
        return ans

    def level_travel2(self, root):  # 递归方式
        depth = 0
        result= []
        def dfs(root, depth, result):
            if not root:
                return
            if len(result)==depth:
                result.append([])

            result[depth].append(node.val)
            dfs(root.left, depth+1, result)
            dfs(root.right, depth+1, result)
        dfs(root, depth, result)
        return result

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or (root.val==p.val) or (root.val==q.val) :
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right= self.lowestCommonAncestor(root.right, p, q)
        if not left:
            return right
        if not right:
            return left
        return root

class Solution:
    def buildTree(self, preorder, inorder):
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        stack = [root]

        index = 0
        for i in range(1, len(preorder)):
            node = TreeNode(preorder[i])
            if stack and stack[-1].val != inorder[index]:
                stack[-1].left = node
                stack.append(node)
            else:
                _root = stack[-1]
                while stack and stack[-1].val == inorder[index]:
                    _root = stack.pop(-1)
                    index += 1
                _root.right = node
                stack.append(node)

        return root

