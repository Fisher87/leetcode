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
    def canpartition(self, nums):
        if len(nums)<2:
            return False
        sums = sum(nums)
        if sums&1:
            return False
        # nums.sort()

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
        
        dp = [ [0]*(n+1) for _ in range(m)]

        for i in range(1, m+1):
            dp[i][0] = i
        for j in range(1, n+1):
            dp[0][j] = j
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
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
        maxlen, max_s = 0, ''
        n = len(s)
        for i in range(1, n):
            left, right = i-1, i
            while right<n and s[right]==s[i]:
                right += 1

            while i>=0 and right<n and s[right]==s[left]:
                left -= 1
                right+= 1

            _maxlen = right-(left+1)
            if _maxlen>maxlen:
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
        n = len(nums)
        bit = Bit(n)
        _nums = sorted(nums)
        for i in range(n-1, -1, -1):
            _id = self.get_id(_nums, nums[i])
            ans += bit.query(_id-1)
            bit.update(_id)
        return ans

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

class MaximalSquare:
    # 计算最大矩阵面积
    def maximalSquare(self, matrix):
        m, n = len(matrix), len(matrix[0])
        dp= [ [0]*n for _ in range(m) ]

        for i in range(m):
            if matrix[i][0] == "1":
                dp[i][0] = 1
        for j in range(n):
            if matrix[0][j] == "1"
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
            _max = = max(left, right)
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

class NumTrees:
    # 不同的二叉搜索树
    def numTree(self, nums):
        n = len(nums)
        dp = [0]*(n+1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n):
            for j in range(1, i+1):
                dp[i] = dp[j-1]*dp[i-j]
        return dp[n]

class MaxProfit:
    # 买卖股票最佳时间
    def maxProfit(self, prices):
        ans = 0
        for i in range(1, len(prices)):
            if prices[i]>prices[i-1]:
                ans += prices[i]-prices[i-1]
            else:
                continue
        return ans

class FindLength:
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
    # 划分为k个相等的子集
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
            if left<=0:
                right += 1
            elif right>=(n-1):
                left -= 1
            elif x-nums[left] <= nums[right]-x:
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
                return break
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
        # 使用层序遍历
        if not root:
            return ''
        return str(root.val) + ',' + self.serialize(root.left) + \
                ',' + self.serialize(root.right)
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

class ReverseBetween:
    def reverseBetween(self, head, left, right):
        # 头插法
        dummy_node = ListNode(-1)
        dummy_node.next = head
        pre = dummy_node
        for i in range(left-1):
            pre = pre.next

        cur = pre.next
        for i in range(right-left):
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
                if nums[0]<target<nums[mid]:
                    right = mid+1
                else:
                    left = mid-1
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
        T = len(obs)
        N = len(states)

        # 初始化Viterbi矩阵和路径矩阵
        viterbi_mat = np.zeros((N, T))
        path_mat = np.zeros((N, T), dtype=int)

        # 初始状态概率
        viterbi_mat[:, 0] = start_prob * emission_prob[:, obs[0]]
        path_mat[:, 0] = 0

        for t in range(1, T):
            for s in range(N):
                prob = viterbi_mat[:, t-1] * trans_prob[:, s] * emission_prob[s, obs[t]]
                viterbi_mat[s, t] = np.max(prob)
                path_mat[s,t] = np.argmax(prob)

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

"""
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

