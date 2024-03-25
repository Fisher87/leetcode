/*================================================================
*   Copyright (C) 2023 Fisher. All rights reserved.
*   
*   文件名称：algorithm.cpp
*   创 建 者：YuLianghua
*   创建日期：2023年08月01日
*   描    述：
*  deque< int > 双向队列
*  queue< int > 队列
*  vector< int > 数组
*  priority_queue< int > 升序队列
*  unordered_map< string, string > 字典
================================================================*/
#include <algorithm>
#include <climits>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    string reverseWords(string s) {
        reverse(s.begin(), s.end());

        int n = s.size();
        int idx = 0;
        for(int start=0; start<n; ++start) {
            if( s[start] != ' ' ) {
                if(idx!=0) s[idx++] = ' ';

                int end=start;
                while(end<n && s[end]!=' '){
                    s[idx++] = s[end++];
                }
                reverse(s.begin()+idx-(end-start), s.begin()+idx);

                start=end;
            }
        }
        s.erase(s.begin()+idx, s.end());
        return s;
    }
};

class FindDuplicate{
public:
    int findDuplicate(std::vector<int>& nums) {
        int slow=0, fast=0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while( slow!=fast );
        slow = 0;
        while (slow!=fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return fast;
    }

};

class CombinationSum {
private:
    vector<int> candidates;
    vector<int> path;
    vector< vector<int> > res;
public:
    void DFS(int start, int target) {
        if (target==0) {
            res.push_back(path);
            return ;
        }

        for (int i=start; i<candidates.size() && target-candidates[i]>=0; i++) {
            if (i>start && candidates[i] == candidates[i-1])
                continue;
            path.push_back(candidates[i]);

            DFS(start+1, target-candidates[i]);
            path.pop_back();
        }
    }

    vector< vector<int> > combinationsum(vector<int> &candidates, int target) {
        sort(candidates.begin(), candidates.end());
        this->candidates = candidates;
        DFS(0, target);
        return res;
    }
};

class RemoveDuplicate {
public:
    string removeDuplicate(string &s) {
        string stk;
        for (char ch:s) {
            if (!stk.empty() && stk.back()==ch) {
                stk.pop_back();
            } else {
                stk.push_back(ch);
            }
        }
        return stk;
    }
};

class SimplifyPath{
public:
    string simplifyPath(string path) {
        auto split = [](const string& s, char delim) -> vector<string> {
            vector<string> ans;
            string cur;
            for (char ch: s) {
                if (ch == delim) {
                    ans.push_back(move(cur));
                    cur.clear();
                }
                else {
                    cur += ch;
                }
            }
            ans.push_back(move(cur));
            return ans;
        };

        vector<string> names = split(path, '/');
        vector<string> stack;
        for (string& name: names) {
            if (name == "..") {
                if (!stack.empty()) {
                    stack.pop_back();
                }
            }
            else if (!name.empty() && name != ".") {
                stack.push_back(move(name));
            }
        }
        string ans;
        if (stack.empty()) {
            ans = "/";
        }
        else {
            for (string& name: stack) {
                ans += "/" + move(name);
            }
        }
        return ans;
    }
};

class Partition{
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* small = new ListNode(0);
        ListNode* smallHead = small;
        ListNode* large = new ListNode(0);
        ListNode* largeHead = large;
        while head!=nullptr {
            if (head->val < x) {
                small->next = head->next;
                small = small->next;
            } else {
                large->next = head->next;
                large = large->next;
            }
            head = head->next;
        }
        large->next = nullptr;
        small->next = largeHead->next;

        return smallHead->next;
    }
};

class LongestPath{
public:
    static constexpr int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int rows, columns;

    int longestIncreasingPath(vector< vector<int> > &matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return 0;
        }
        rows = matrix.size();
        columns = matrix[0].size();
        auto memo = vector< vector<int> > (rows, vector <int> (columns));
        int ans = 0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                ans = max(ans, dfs(matrix, i, j, memo));
            }
        }
        return ans;
    }

    int dfs(vector< vector<int> > &matrix, int row, int column, vector< vector<int> > &memo) {
        if (memo[row][column] != 0) {
            return memo[row][column];
        }
        ++memo[row][column];
        for (int i = 0; i < 4; ++i) {
            int newRow = row + dirs[i][0], newColumn = column + dirs[i][1];
            if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns && matrix[newRow][newColumn] > matrix[row][column]) {
                memo[row][column] = max(memo[row][column], dfs(matrix, newRow, newColumn, memo) + 1);
            }
        }
        return memo[row][column];
    }
};

class FourSum {
public:
    vector< vector<int> > fourSum(vector<int>& nums, int target){
        vector<vector<int>> quadruplets;
        if (nums.size()<4){
            return quadruplets;
        }

        sort(nums.begin(), nums.end());
        int length = nums.size();
        for (int i=0; i<length-3; ++i) {
            while (i>0 && nums[i]==nums[i-1]) {
                continue;
            }
            if((long) nums[i]+nums[i+1]+nums[i+2]+nums[i+3]>target) {
                break;
            }
            if((long) nums[i]+nums[length-3]+nums[length-2]+nums[length-1]<target) {
                continue;
            }

            for (int j=i+1; j<length-2; ++j) {
                while (j>i+1 && nums[j]==nums[j-1]) {
                    continue;
                }
                if((long) nums[i]+nums[j]+nums[j+1]+nums[j+2] > target) {
                    break;
                }
                if((long) nums[i]+nums[j]+nums[length-2]+nums[length-1] < target) {
                    continue;
                }
                int left = j+1, right = length-1;
                while (left<right) {
                    long sum = (long) nums[i]+nums[j]+nums[left]+nums[right];
                    if (sum==target) {
                        quadruplets.push_back({nums[i], nums[j], nums[left], nums[right]});
                        while (left<right && nums[left]==nums[left+1]) {
                            left ++;
                        }
                        left ++;
                        while(left<right && nums[right]==nums[right-1]){
                            right--;
                        }
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        right--;
                    }
                }

            }

        }

    }

};

class MyStack {
public:
    queue<int> q;
    MyStack(){}

    void push(int x) {
        int n = q.size();
        q.push(x);
        for (int i=0; i<n; i++) {
            q.push(q.front());
            q.pop();
        }
    }

    int pop() {
        int r = q.front();
        q.pop();
        return r;
    }

    int top(){
        int r = q.front();
        return r;
    }

    bool empty(){
        return q.empty();
    }

};

class RecoverTree{
public:
    void inorder(TreeNode* root, vector<int>& nums) {
        if (root == nullptr) {
            return;
        }
        inorder(root->left, nums);
        nums.push_back(root->val);
        inorder(root->right, nums);
    }

    pair<int,int> findTwoSwapped(vector<int>& nums) {
        int n = nums.size();
        int index1 = -1, index2 = -1;
        for (int i = 0; i < n - 1; ++i) {
            if (nums[i + 1] < nums[i]) {
                index2 = i + 1;
                if (index1 == -1) {
                    index1 = i;
                } else {
                    break;
                }
            }
        }
        int x = nums[index1], y = nums[index2];
        return {x, y};
    }

    void recover(TreeNode* r, int count, int x, int y) {
        if (r != nullptr) {
            if (r->val == x || r->val == y) {
                r->val = r->val == x ? y : x;
                if (--count == 0) {
                    return;
                }
            }
            recover(r->left, count, x, y);
            recover(r->right, count, x, y);
        }
    }

    void recoverTree(TreeNode* root) {
        vector<int> nums;
        inorder(root, nums);
        pair<int,int> swapped= findTwoSwapped(nums);
        recover(root, 2, swapped.first, swapped.second);
    }
};

class LargestRectangArea {
public:
    int largestRectangleArea(vector<int> & heights) {
        int n = heights.size();
        vector<int> left(n), right(n);

        stack<int> mono_stack;
        for (int i=0; i<n; ++i) {
            while (!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]) {
                mono_stack.pop();
            }
            left[i] = (mono_stack.empty() ? -1 : mono_stack.top());
            mono_stack.push(i);
        }

        mono_stack = stack<int>();
        for(int i=n-1; i>=0; --i) {
            while (!mono_stack.empty() && heights[mono_stask.top()] >= heights[i]) {
                mono_stack.pop();
            }
            right[i] = (mono_stack.empty() ? n : mono_stack.top());
            mono_stack.push(i);
        }

        int ans = 0;
        for(int i=0; i<n; ++i) {
            ans = max(ans, (right[i]-left[i]-1) * heights[i]);
        }
    }

};

/*
 * 组合数
 */
class Permutation{
public:
    vector<string> rec;
    vector<int> vis;
    void backtrack(const string& s, int i, int n, string& perm) {
        if (i == n) {
            rec.push_back(perm);
            return;
        }
        for (int j = 0; j < n; j++) {
            if (vis[j] || (j > 0 && !vis[j - 1] && s[j - 1] == s[j])) {
                continue;
            }
            vis[j] = true;
            perm.push_back(s[j]);
            backtrack(s, i + 1, n, perm);
            perm.pop_back();
            vis[j] = false;
        }
    }

    vector<string> permutation(string s) {
        int n = s.size();
        vis.resize(n);
        sort(s.begin(), s.end());
        string perm;
        backtrack(s, 0, n, perm);
        return rec;
    }
};

/*
 * 三数之和为0
 */
class ThreeSum{
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector< vector<int> > ans;

        for (int i=0; i<n; i++) {
            if (i>0 && nums[i]==nums[i-1]) continue;
            int k = n-1;
            int target = -nums[i];

            for (int j=i+1; j<n; j++) {
                if (j>i+1 && nums[j]==nums[j-1]) continue;

                while (j<k && nums[j]+nums[k] > target) {
                    --k;
                }
                if (j==k) break;

                if (nums[j]+nums[k]==target){
                    ans.push_back({nums[i], nums[j], nums[k]});
                }
            }
        }
        return ans;
    }
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode():val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x): val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right): val(x), left(left), right(right) {}
};
/* 前序与中序构建二叉树  */
class BuildTree{
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if (!preorder.size()) {
            return nullptr;
        }
        TreeNode* root = new TreeNode(preorder[0]);
        stack<TreeNode*> stk;
        stk.push(root);
        int inorderIndex = 0;
        for (int i=1; i<preorder.size(); ++i) {
            int preorderVal = preorder[i];
            TreeNode* node = stk.top();
            if (node->val != inorder[inorderIndex]) {
                node->left = new TreeNode(preorderVal);
                stk.push(node->left);
            } else {
                while (!stk.empty() && stk.top()->val==inorder[inorderIndex]) {
                    node = stk.top();
                    stk.pop();
                    ++inorderIndex;
                }
                node->right = new TreeNode(preorderVal);
                stk.push(node->right);
            }
        }
        return root;
    }
};

/* 排序链表 */
class SortList {
public:
    ListNode* sortlist(ListNode* head) {
        if (!head || !head->next) {
            return head;
        }

        ListNode *slow = head, *fast = head;
        while(fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        fast = slow;
        slow = slow->next;
        fast->next = nullptr;

        return mergeTwoLists(sortlist(head), sortlist(slow));
    }

    ListNode* merge(ListNode *l1, ListNode *l2) {
        ListNode dummynode{0};
        auto curr = &dummynode;

        while (l1 && l2) {
            if (l1->val <= l2->val) {
                curr->next = l1;
                l1 = l1->next;
            } else {
                curr->next = l2;
                l2 = l2->next;
            }
            curr = curr->next;
        }
        curr->next = l1 ? l1 : l2;

        return dummynode.next;
    }
};

class MaxSlidingWindow{
public:
    maxslidingWindow(vector<int>& nums, int k) {
        vector<int> result;
        deque<int> dq;
        for(int i=0; i<nums.size(); i++) {
            while (!dq.empty() && i-dq.front()>=k) {
                dq.pop_front();
            }
            while(!dq.empty() && nums[dq.back()]<=nums[i]) {
                dq.pop_back();
            }
            dq.emplace_back(i);
            if (i>=k-1) {
                result.emplace_back(nums[dq.front()]);
            }
        }
        return result;
    }

};

// 判断括号是否有效
class IsValid{
public: 
    bool isValid(string s) {
        unordered_map<char, int> m{ {'(', 1}, {'[', 2}, {'{', 3} 
                                    {')', 4}, {']', 5}, {'}', 6}};
        stack<char> st;
        bool istrue = false;

        for (char c:s) {
            int flag = m[c];
            if (flag>=1 && flag<=3) {
                st.push(c);
            } else if (!=st.empty() && m[st.top()]==flag-3) {
                st.pop();
            } else {
                istrue = false;
                break;
            }
        }
        if (!st.empty()) {
            istrue = false;
        }
        return istrue;
    }
};

// 最多元素;
class MajorityElement {
public:
    int majorityElement(vector<int>& nums) {
        int ele = nums[0];
        int cnt = 1;
        for (int i=1; i<nums.size(); i++) {
            if (cnt==0) {
                ele = nums[i];
                cnt = 1;
                continue;
            }
            if (nums[i]==ele) {
                ++cnt;
            } else {
                --cnt;
            }
        }
        return ele;
    }
};

// 最长回文子串
class LongestPalidrome{
    public:
        string longestPalidrome(string s) {
           int n = s.size(); 
           if (n<=1) return s;
           int maxlen=0;
           string maxstr = "";
           for(int i=0; i<n; ++i) {
               int left=i-1, right=i;
               while(right<n && s[right]==s[i]) {
                   right++;
               } 
               while(left>=0 && right<n && s[left]==s[right]) {
                   left--;
                   right++;
               }
               int _len = right-(left+1);
               if (_len > maxlen) {
                   maxlen = _len;
                   maxstr = s.substr(left+1, right-(left+1));
               }
           }

           return maxstr;
        }
};

// 路径数
class UniquePathsWithObstacles {
public:
    int uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();

        vector< vector<int> > dp(m, vector<int>(n, 0));
        dp[0][0] = (obstacleGrid[0][0]==1) ? 0 : 1;

        for(int i=1; i<m; ++i) {
            dp[i][0] = (obstacleGrid[i][0]==1) ? 0 : dp[i-1][0];
        }

        for (int j=1; j<n; ++j) {
            dp[0][j] = (obstacleGrid[0][j]==1) ? 0 : dp[0][j-1];
        }

        for(int i=1; i<m; i++) {
            for (int j=1; j<n; j++) {
                if (obstacleGrid[i][j]==1) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }

        return dp[m-1][n-1];
    }
};


// LRU cache
#include <iostream>
#include <unordered_map>

class DLinkedNode {
public:
    int key;
    int value;
    DLinkedNode* prev;
    DLinkedNode* next;
    
    DLinkedNode(int _key=0, int _value=0) : key(_key), value(_value), prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    std::unordered_map<int, DLinkedNode*> cache;
    DLinkedNode* head;
    DLinkedNode* tail;
    int capacity;
    int size;

public:
    LRUCache(int _capacity) : capacity(_capacity), size(0) {
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head->next = tail;
        tail->prev = head;
    }

    ~LRUCache() {
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            delete it->second;
        }
        delete head;
        delete tail;
    }

    int get(int key) {
        if (cache.find(key) == cache.end()) {
            return -1;
        }

        DLinkedNode* node = cache[key];
        moveToHead(node);
        return node->value;
    }

    void put(int key, int value) {
        if (cache.find(key) == cache.end()) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode* node = new DLinkedNode(key, value);
            // 添加进哈希表
            cache[key] = node;
            // 添加至双向链表的头部
            addToHead(node);
            size++;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode* removed = removeTail();
                // 删除哈希表中对应的项
                cache.erase(removed->key);
                size--;
                delete removed;
            }
        } else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            DLinkedNode* node = cache[key];
            node->value = value;
            moveToHead(node);
        }
    }

private:
    void addToHead(DLinkedNode* node) {
        node->prev = head;
        node->next = head->next;
        head->next->prev = node;
        head->next = node;
    }

    void removeNode(DLinkedNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void moveToHead(DLinkedNode* node) {
        removeNode(node);
        addToHead(node);
    }

    DLinkedNode* removeTail() {
        DLinkedNode* node = tail->prev;
        removeNode(node);
        return node;
    }
};


// 排序算法
class SortArray {
public:
    vector<int> heapSort(vector<int>& nums) {
        heapsort(nums);
        return nums;
    }

    vector<int> quickSort(vector<int>& nums) {
        quick_sort(nums, 0, nums.size()-1);
        return nums;
    }

    vector<int> mergeSort(vector<int>& nums) {
        mergesort(nums, 0, nums.size()-1);
        return nums;
    }

private:
    // begin heap sort
    void heapsort(vector<int>& nums) {
        int n = nums.size();
        for (int i=n/2-1; i>=0; i--) {
            buildHeap(i, n, nums);
        }

        // heap sort
        for (int i=n-1; i>0; i--){
            swap(nums[0], nums[i]);
            buildHeap(0, i, nums);
        }
    }

    void buildHeap(int i, int size, vector<int>& nums){
        int largest = i;
        int left = 2*i+1;
        int right = 2*i+2;
        if (left < size && nums[left]>nums[largest]) {
            largest = left;
        }
        if (right< size && nums[right]>nums[largest]) {
            largest = right;
        }

        if (largest != i) {
            std::swap(nums[i], nums[largest]);
            buildHeap(largest, size, nums);
        }
    }
    // end heap sort

    // begin quick sort
    void quick_sort(vector<int>& nums, int left, int right) {
        if (left>=right) {
            return ;
        }
        int pivot = partition(nums, left, right);
        quick_sort(nums, left, pivot-1);
        quick_sort(nums, pivot+1, right);
    }
    int partition(vector<int>& nums, int left, int right) {
        int pivotIndex = rand() % (right-left+1) + left;
        std::swap(nums[pivotIndex], nums[right]);

        int i=left;
        for (int j=left; j<right; ++j) {
            if (nums[j] < nums[right]) {
                std::swap(nums[i], nums[j]);
                i++;
            }
        }
        std::swap(nums[i], nums[right]);
        return i;
    }
    // end quick sort
    
    // begin merge sort
    void mergesort(vector<int>& nums, int left, int right) {
        if (left>=right) {
            return;
        }
        int mid = left + (right-left)/2;
        mergesort(nums, left, mid);
        mergesort(nums, mid+1, right);
        int i=left, j=mid+1;
        std::vector<int> tmp;
        while(i<=mid || j<=right) {
            if (i>mid || (j<=right && nums[i]<nums[j])) {
                tmp.push_back(nums[j]);
                j++;
            } else {
                tmp.push_back(nums[i]);
                i++;
            }
        }

        for (int k=left; k<=right; k++) {
            nums[k] = tmp[k-left];
        }
    }
    // end merge sort
};

// 相交链表节点
class GetIntersectionNode {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA==nullptr || headB==nullptr) {
            return nullptr;
        }
        ListNode* p = headA;
        ListNode* q = headB;
        while (p!=q) {
            p = p ? p->next : headB;
            q = q ? q->next : headA;
        }
        return p;
    }
};


// 路径和 II
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        ans.clear();
        vector<int> path;
        dfs(root, target, path, ans);
        return ans;
    }
private:
    vector< vector<int> > ans;
    void dfs(TreeNode* root, int target, vector<int>& path, vector< vector<int> >& ans) {
        if (root==nullptr) {
            return;
        }
        if (root->left==nullptr && root->right==nullptr && target==root->val) {
                path.push_back(root->val);
                ans.push_back(path);
                path.pop_back(root->val);
                return ;
            }
        }
        path.push_back(root->val);
        dfs(root->left, target-root->val, path, ans);
        dfs(root->right, target-root->val, path, ans);
        path.pop_back();
    }
};


// 实现trie 前缀树
class Trie {
private:
    bool isWord;
    std::unordered_map<char, Trie*> children;

public:
    Trie() : isWord(false) {}

    void insert(const std::string& word) {
        Trie* node = this;
        for (char w : word) {
            if (node->children.find(w) == node->children.end()) {
                node->children[w] = new Trie();
            }
            node = node->children[w];
        }
        node->isWord = true;
    }

    Trie* searchPrefix(const std::string& prefix) {
        Trie* node = this;
        for (char w : prefix) {
            if (node->children.find(w) == node->children.end()) {
                return nullptr;
            }
            node = node->children[w];
        }
        return node;
    }

    bool search(const std::string& word) {
        Trie* node = searchPrefix(word);
        return node != nullptr && node->isWord;
    }

    bool startsWith(const std::string& prefix) {
        Trie* node = searchPrefix(prefix);
        return node != nullptr;
    }

    // Destructor to deallocate memory
    ~Trie() {
        for (auto& child : children) {
            delete child.second;
        }
    }
};

// 对称二叉树
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(root == nullptr) {
            return true;
        }
        return check(root->left, root->right);
    }
private:
    bool check(TreeNode* left, TreeNode* right) {
        if (left==nullptr && right==nullptr) { return true; } 
        if (left==nullptr || right==nullptr) { return false;}

        if (left->val!=right->val) {
            return false;
        } else {
            return check(left->right, right->left) && check(left->left, right->right);
        }
    }
};

// 编辑距离
class EditDistance {
public:
    int editDistance(string word1, string word2) {
        int m=word1.size(), n=word2.size();
        if (m==0 || n==0) {
            return m ? n==0 : n ;
        }

        vector< vector<int> > dp(m+1, vector<int>(n+1));
        for (int i=0; i<m+1; i++) {
            dp[i][0] = i;
        }
        for (int j=0; j<n+1; j++) {
            dp[0][j] = j;
        }

        for (int i=1; i<m+1; i++) {
            for (int j=1; j<n+1; j++) {
                if (word1[i-1]==word2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = min( dp[i][j-1], min(dp[i-1][j], dp[i-1][j-1]) ) + 1;
                }
            }
        }
        return dp[m][n];
    }

};

// 全排列
class Permute {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        ans.clear();
        int size = nums.size();
        vector< bool > visited(size);
        vector<int> path;
        backtrack(nums, path, 0, size, visited);
        return ans;
    }
private:
    vector< vector<int> > ans;
    void backtrack(vector<int>& nums, vector<int>& path, int depth, int size, vector<bool>& visited){
        if(depth==size){
            ans.push_back(path);
            return ;
        }
        for (int i=0; i<size; i++) {
            if (visited[i]) {
                continue;
            }
            path.push_back(nums[i]);
            visited[i] = true;
            backtrack(nums, path, depth+1, size, visited);
            visited[i] = false;
            path.pop_back();
        }
    }
};

// 最大子数和
class MaxSubArray {
public:
    int maxSubArray(vector<int>& nums) {
        int presum = nums[0];
        int maxsum = nums[0];
        for (int i=1; i<nums.size(); i++) {
            presum = max(presum+nums[i], nums[i]);
            maxsum = max(presum, maxsum);
        }
        return maxsum;
    }
};

// 验证二叉搜索树 
class IsValidBST {
public:
    bool isValidBST(TreeNode* root) {
        return isvalid(root, LONG_MIN, LONG_MAX);
    }
private:
    bool isvalid(TreeNode* root, long long low, long long high) {
        if (root == nullptr) {
            return true;
        }
        if (root->val > low && root->val < high) {
            return isvalid(root->left, low, root->val) && isvalid(root->right, root->val, high);
        }
        return false;
    }
};

// 二叉树层序遍历 
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        ans.clear();
        stack.clear();
        if (root==nullptr) {
            return ans;
        }
        stack.emplace_back(root);
        while (!stack.empty()) {
            vector<int> tmp;
            int size = stack.size();
            for (int i=0; i<size; i++) {
                TreeNode* node = stack.front();
                stack.pop_front();
                tmp.push_back(node->val);
                if (node->left!=nullptr) {
                    stack.emplace_back(node->left);
                }
                if (node->right!=nullptr) {
                    stack.emplace_back(node->right);
                }
            }
            ans.push_back(tmp);
        }
        return ans;
    }
private:
    deque<TreeNode*> stack;
    vector< vector<int> > ans;
};

// 数组中第k个最大元素
class FindKthLargest {
private:
    void maxHeapify(vector<int>& nums, int i, int size) {
        int left = 2*i + 1;
        int right = 2*i + 2;
        int largest = i;
        if (left<size && nums[largest]<nums[left]) {
            largest = left;
        }
        if (right<size && nums[largest]<nums[right]) {
            largest = right;
        }
        if (largest != i) {
            swap(nums[largest], nums[i]);
            maxHeapify(nums, largest, size);
        }
    }

    void buildHeap(vector<int>& nums, int size) {
        for (int i=size/2; i>=0; i--) {
            maxHeapify(nums, i, size);
        }
    }

public:
    int findKthLargest(vector<int>& nums, int k) {
        int n = nums.size();
        buildHeap(nums, n);
        
        for (int i=n-1; i>=(n-1-k); i++) {
            swap(nums[0], nums[i]);
            maxHeapify(nums, 0, i);
        }
        return nums[0];
    }

    int findKthLargestv1(vector<int>& nums, int k) {
        std::priority_queue<int, vector<int>, std::greater<int> > minheap; // 使用优先队列
        for (i=0; i<k; i++) {
            minheap.push(nums[i]);
        }

        for(size_t i=k; i<nums.size(); i++) {
            int kth = minheap.top();
            int _kth= kth>nums[i] ? kth : mins[i];
            minheap.pop();
            minheap.push(_kth);
        }
        return minheap.top();
    }

    int findKthLargest2(vector<int>& nums, int k) {
        make_heap(nums.begin(), nums.end()); // 创建堆
        for(int i =0;i<k-1;i++){
            pop_heap(nums.begin(),nums.end());
            nums.pop_back();
        }
        return nums[0];
    }
};

// 判断是否是回文链表
class IsPalindromeList {
public:
    bool isPalindrome(ListNode* head) {
        int size = 0;
        ListNode* cur = head;
        while(cur!=nullptr) {
            cur = cur->next;
            size++;
        }
        if (size<=1) {
            return true;
        }
        ListNode* slow = head;
        ListNode* fast = head->next;
        while (fast!=nullptr && fast->next!=nullptr) {
            slow = slow->next;
            fast = fast->next->next;
        }
        std::cout << slow->val << endl;
        ListNode* _l1 = head;
        ListNode* l2 = slow->next;
        slow->next = nullptr;
        ListNode* l1 = reverseList(_l1);
        if ((size)%2) {
            l1 = l1->next;
        }
        while (l1!=nullptr && l2!=nullptr) {
            if (l1->val != l2->val) {
                return false;
            } else {
                l1 = l1->next;
                l2 = l2->next;
            }
        }
        return true;
    }
private:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur!=nullptr) {
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
};

// Z 打印
class zigzagLevelOrder{
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        if(root==nullptr){
            return ans;
        }
        queue<TreeNode*> dq;
        dq.push(root);
        bool l2r = true;
        while(!dq.empty()){
            size_t size = dq.size();
            vector<int> tmp;
            for(int i=0; i<size; i++) {
                TreeNode* node = dq.front();
                tmp.push_back(node->val);
                if(node->left!=nullptr) {
                    dq.push(node->left);
                }
                if(node->right!=nullptr) {
                    dq.push(node->right);
                }
                dq.pop();
            }
            if(not l2r){
                std::reverse(tmp.begin(), tmp.end());
            }
            l2r = !l2r;
            ans.push_back(tmp);
        }
        return ans;
    }
private:
    vector<vector<int>> ans;
};

// 最近公共祖先
class LowestCommonAncestor {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        bool ret = check(root, p, q);
        return ans
    }
private:
    TreeNode* ancestor;
    bool check(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (root == nullptr) {
            return false;
        }
        bool flag1 = check(root->left, p, q);
        bool flag2 = check(root->right, p, q);
        bool flag3 = ((root==p) || (root==q));
        if ((flag1&&flag2) || (flag1||flag2) && flag3) {
            ans = root;
        }
        if (flag1 || flag2 || flag3) {
            return true;
        } 
        return false;
    }
};

class LowestCommonAncestor {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root->val==p->val || root->val==q->val) {
            return root;
        }
        TreeNode* leftancestor = lowestCommonAncestor(root->left, p, q);
        TreeNode* rightancestor= lowestCommonAncestor(root->right, p, q);
        if (!leftancestor) {
            return rightancestor;
        }
        if (!rightancestor) {
            return leftancestor;
        }
        return root;
    }
};

// 岛屿数量
class NumIsland {
public:
    int numIslands(vector<vector<char>>& grid) {
        size_t m = grid.size();
        size_t n = grid[0].size();
        int ans = 0;
        for (size_t i=0; i<m; i++) {
            for (size_t j=0; j<n; j++) {
                if (grid[i][j]=='1') {
                    ans++;
                    dfs(grid, i, j, m, n);
                }
            }
        }
        return ans;
    }

private:
    void dfs(vector<vector<char>>& grid, int i, int j, int m, int n) {
        if (i<0 || i>=m || j<0 || j>=n) {
            return ;
        } 
        if (grid[i][j]!='1') {
            return ;
        } 
        grid[i][j] = '2';
        for (auto& [di,dj] : directs) {
            size_t _i = i+di;
            size_t _j = j+dj;
            dfs(grid, _i, _j, m, n);
        }

    }
    vector<pair<int, int>> directs{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
};

// 最长上升子序列 
class LengthOfLIS {
public:
    int lengthOfLIS(vector<int>& nums) {
        size_t n = nums.size();
        vector<int>::iterator ans;
        vector<int> dp(n, 1);
        for (size_t i=1; i<n; i++) {
            for (size_t j=0; j<i; j++) {
                if (nums[i]>nums[j]) {
                    dp[i] = max(dp[i], dp[j]+1);
                }
            }
        }
        ans = max_element(dp.begin(), dp.end());
        // auto ans = max_element(dp.begin(), dp.end());
        return *ans;
    }
};

// 搜索旋转排序数组
class Search {
public:
    int search(vector<int>& nums, int target) {
        size_t n = nums.size();
        int left=0, right=n-1;
        while (left<=right) {
            size_t mid = (left+right)/2;
            if (nums[mid]==target) {
                return mid;
            }
            // 总结一点：找到绝对递增的部分
            if (nums[0]<=nums[mid]) {
                if(nums[mid]>target && target>=nums[0]) {
                    right = mid-1;
                } else {
                    left = mid+1;
                }
            } else {
                if (nums[mid]<target && target<=nums[n-1]) {
                    left = mid+1;
                } else {
                    right= mid-1;
                }
            }
        }
        return -1;
    }
};

// x 的平方根
class MySqrt {
public:
    int mySqrt(int x) {
        int left = 0;
        int right= x;
        while (left <= right) {
            int mid = (left+right)/2;
            if (mid*mid<=x && (mid+1)*(mid+1)>x) {
                return mid;
            } else if (mid*mid > x){
                right = mid-1;
            } else {
                left = mid+1;
            }
        }
        return left+1;
    }
};

// 无重复字符最长子串
class LengthOfLongestSubstring{
public:
    int lengthOfLongestSubstring(string s) {
        deque<char> window;
        int maxlen = 0;
        for(auto c: s) {
            while(std::find(window.begin(), window.end(), c)!=window.end()){
                window.pop_front();
            }
            window.push_back(c);
            maxlen = max(maxlen, int(window.size()));
        }
        return maxlen;
    }
};

// 合并K个升序链表
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class MergeKLists {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto compare = [](const std::pair<int, int>&a, const std::pair<int, int>& b) {
            return a.first > b.first;
        };
        std::priority_queue< std::pair<int, int>, std::vector<std::pair<int, int>>, decltype(compare) > pq;
        for(int i=0; i<lists.size(); i++) {
            if (lists[i]!=nullptr) {
                pq.push({lists[i]->val, i});
            }
        }
        ListNode* dummynode = new ListNode(-1);
        ListNode* cur = dummynode;
        while (!pq.empty()) {
            auto [val, idx] = pq.top();
            pq.pop();
            cur->next = lists[idx];
            cur = cur->next;
            lists[idx] = lists[idx]->next;

            if (lists[idx] != nullptr) {
                pq.push({lists[idx]->val, idx});
            }
        }
        ans = dummynode->next;
        delete dummynode;

        return ans;
    }
private:
    ListNode* ans;
};

// 查找两个正序数组中位数
class FindMedianSortedArrays {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m=num1.size(), n=nums2.size();
        if ((m+n)%2) {
            int mid_k = (m+n)/2 + 1;
            ans = get_k(nums1, nums2, mid_k);
        } else {
            int mid_l = (m+n)/2;
            int mid_r = (m+n)/2 + 1;
            ans = (get_k(nums1, nums2, mid_l)+get_k(nums1, nums2, mid_r)) * 0.5;
        }
        return ans;
    }

private:
    double get_k(vector<int>& nums1, vector<int>& nums2, int k) {
        int index1 = 0;
        int index2 = 0;
        while (true) {
            if (index1==nums1.size()) {
                return nums2[index2+k-1];
            } else if (index2==nums2.size()) {
                return nums1[index1+k-1];
            } else if (k==1) {
                return std::min(nums1[index1], nums2[index2]);
            }

            int _index1 = min(index1+k/2-1, int(nums1.size()-1));
            int _index2 = min(index2+k/2-1, int(nums2.size()-1));
            if (nums1[_index1] <= nums2[_index2]) {
                k = k - (_index1-index1+1);
                index1 = _index1+1;
            } else {
                k = k - (_index2-index2+1);
                index2 = _index2+1;
            }
        }
    }
    double ans ;
};

// 螺旋遍历二维数组
class SpiralArray {
public:
    vector<int> spiralArray(vector<vector<int>>& array) {
        if (array.empty()) {
            return vector<int> {};
        }
        int m = array.size(); 
        int n = array[0].size();
        int left=0, right=n-1, top=0, bottom=m-1;
        int total_count = m*n, count = 0;
        vector<int> ans;
        while (count < total_count) {
            if(left>right) { break; }
            for (int i=left; i<=right; i++) {
                ans.push_back(array[top][i]);
                count++;
            }
            top++;

            if(top > bottom) { break; }
            for (int i=top; i<=bottom; i++) {
                ans.push_back(array[i][right]);
                count++;
            }
            right--;

            if (right<left) { break; }
            for(int i=right; i>=left; i--) {
                ans.push_back(array[bottom][i]);
                count++;
            }
            bottom--;

            if (bottom<top) { break; }
            for(int i=bottom; i>=top; i--) {
                ans.push_back(array[i][left]);
                count++;
            }
            left++;
        }
        return ans;
    }
};

// 最大岛屿面积
class MaxAreaOfIsland {
public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int maxarea=0;
        int m=grid.size();
        int n=grid[0].size();

        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                if (grid[i][j]==1) {
                    int ans = dfs(grid, i, j, m, n);
                    maxarea = max(ans, maxarea);
                }
            }
        }
        return maxarea;
    }
private:
    int dfs(vector< vector<int> >& grid, int i, int j, int m, int n) {
        if(i<0 || i>=m || j<0 || j>=n) {
            return 0;
        }
        if (grid[i][j] != 1) {
            return 0;
        }
        grid[i][j] = 2;
        int ans = 1;
        for (auto& [di, dj]: vector<pair<int, int>> { {0,1}, {0,-1}, {1,0}, {-1,0} }) {
            int _i = i+di;
            int _j = j+dj;
            ans += dfs(grid, _i, _j, m, n);
        }
        return ans;
    }
};


// 搜索二维矩阵
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m=matrix.size();
        int n=matrix[0].size();
        int i=0, j=n-1;
        while (i>=0 && i<m && j>=0 && j<n) {
            if (matrix[i][j]==target) {
                return true;
            } else if (matrix[i][j] > target) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }
};

// 异位词分组
class GroupAnagrams{
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        auto arrayHash = [fn=hash<int>{}] (const array<int, 26>& arr) -> size_t {
            return accumulate(arr.begin(), arr.end(), 0u, [&](size_t acc, int num){
                              return (acc<<1)^fn(num);
                              });

        };

        unordered_map<array<int, 26>, vector<string>, decltype(arrayHash)> mp(0, arrayHash);

        for (const string& s: strs) {
            array<int, 26> count{};
            for (char c: s) {
                count[c-'a'] += 1;
            }
            mp[count].emplace_back(s);
        }
        vector< vector<string> > result;
        for (auto& e: mp) {
            result.push_back(e.second);
        }
        return result;
    }
};

// 长度最小的子数组
class MinsubArrayLen {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        queue<int> window;
        int tmp = 0;
        int minsize = INT_MAX;
        for (int num: nums) {
            window.push(num);
            tmp += num;
            while (tmp>=target) {
                minsize = min(minsize, window.size());
                int t = window.front();
                tmp -= t;
                window.pop();
            }
        }
        return minsize<INT_MAX ? minsize : 0;
    }
};

// 最大矩形
class MaximalRectangle{
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if (m==0) {
            return 0;
        }
        int n = matrix[0].size();
        std::vector< vector<int> > dmatrix;
        for (const auto& row : matrix) {
            std::vector<int> int_row;
            for (const auto& cell: row) {
                int_row.push_back(cell-'0');
            }
            dmatrix.push_back(int_row);
        }

        hists = gethists(dmatrix);
        int max_area = 0;
        for(const auto& hist :hists ) {
            max_area = std::max(max_area, get_max(hist));
        }
        return max_area;
    }

    std::vector< vector<int> > gethists(vector< vector<int> >& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        std::vector< std::vector<int> > hists = { matrix[0] };

        for (int i=1; i<m; i++) {
            std::vector<int> hist;
            for (int j=0; j<n; j++) {
                if ( matrix[i][j] == 0 )  {
                    hist.push_back(0);
                } else {
                    hist.push_back(hists.back()[j] + 1);
                }
            }
            hists.push_back(hist);
        }
        return hists;
    }

    int get_max(const std::vector<int> hist) {
        int n = hist.size();
        std::stack<int> stack;
        std::vector<int> left(n, -1);
        std::vector<int> right(n, n);

        for (int i = 0; i < n; ++i) {
            while (!stack.empty() && hist[stack.top()] >= hist[i]) {
                stack.pop();
            }
            if (!stack.empty()) {
                left[i] = stack.top();
            }
            stack.push(i);
        }

        while (!stack.empty()) {
            stack.pop();
        }

        for (int i = n - 1; i >= 0; --i) {
            while (!stack.empty() && hist[stack.top()] >= hist[i]) {
                stack.pop();
            }
            if (!stack.empty()) {
                right[i] = stack.top();
            }
            stack.push(i);
        }

        int max_area = 0;
        for (int i = 0; i < n; ++i) {
            max_area = std::max(max_area, hist[i] * (right[i] - left[i] - 1));
        }
        return max_area;
    }
    }
};

// 寻找重复数, 使用O(1) 空间复杂度;
class FindDuplicate{
public:
    int findDuplicate(vector<int>& nums) {
        int slow=0, fast=0;
        while(true) {
            slow = nums[slow];
            fast = nums[nums[fast]];
            if (slow==fast) {
                break;
            }
        }
        fast = 0;
        while(slow!=fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return fast;
    }
};

// 数组中的重复数据, 
class FindDuplicates{
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> ans;
        for (const auto& num: nums) {
            int x = abs(num);
            if (nums[x-1]>0) {
                nums[x-1] = -nums[x-1];
            } else {
                ans.push_back(x);
            }
        }
        return ans;
    }
};

// 最小路径和
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));

        dp[0][0] = grid[0][0];
        for(int i=1; i<m; i++) {
            dp[i][0] = grid[i][0] + dp[i-1][0];
        }
        for(int j=1; j<n; j++) {
            dp[0][j] = grid[0][j] + dp[0][j-1];
        }

        for (int i=1; i<m; i++) {
            for (int j=1; j<n; j++) {
                dp[i][j] = std::min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
        return dp[m-1][n-1];
    }
};

// 最长回文子串
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        if (n<=1) {
            return s;
        }
        int maxlen = 0;
        string maxstr = "";
        for (int i=0; i<n; i++) {
            int left = i-1, right = i;
            while (right<n && s[right]==s[i]) {
                ++right;
            }
            while (left>=0 && right<n && s[left]==s[right]) {
                --left;
                ++right;
            }
            int tmp = right-(left+1);
            if(tmp>maxlen) {
                maxlen = tmp;
                maxstr = s.substr(left+1, maxlen);
            }
        }
    return maxstr;
    }
};

// 买卖股票最佳时机
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minprice=prices[0];
        int maxprofit=0;
        for (int i=1; i<prices.size(); i++) {
            minprice = std::min(minprice, prices[i]);
            maxprofit = std::max(maxprofit, prices[i]-minprice);
        }
        return maxprofit;
    }
};

// 平衡二叉树,左右子树高度差不超过1；
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        if (root==nullptr) {
            return true;
        }
        int ldepth = depth(root->left);
        int rdepth = depth(root->right);
        return (std::abs(ldepth-rdepth)<=1 && isBalanced(root->left) && isBalanced(root->right));
        
    }
private:
    int depth(TreeNode* root) {
        if(root==nullptr) {
            return 0;
        }
        return std::max(depth(root->left), depth(root->right))+1;
    }
};

// 字符串解码
// k[encoded_string]
class Solution {
public:
    string decodeString(string s) {
        std::stack<string> stack;
        for(char c: s) {
            if (c!=']') {
                stack.push(std::string(1, c));

            } else {
                string num="";
                string _s = "";
                while (!stack.empty()) {
                    string _c = stack.top();
                    stack.pop();
                    if (_c=="[") {
                        break;
                    }
                    _s = _c + _s;
                }
                while (!stack.empty() && isdigit(stack.top()[0])) {
                    num = stack.top() + num;
                    stack.pop();
                }

                std::string ns = "";
                int repeat = std::stoi(num);
                for (int i = 0; i < repeat; ++i) {
                    ns += _s;
                }
                stack.push(ns);
            }
        }
        std::string result = "";
        while (!stack.empty()) {
            result = stack.top() + result;
            stack.pop();
        }

        return result;
    }
};

// 子树
class Solution {
public:
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        if(root==nullptr) {
            return false;
        }
        return sametree(root, subRoot) || isSubtree(root->left, subRoot) ||
             isSubtree(root->right, subRoot);

    }
private:
    bool sametree(TreeNode* root, TreeNode* subRoot) {
        if(root==nullptr && subRoot==nullptr) {
            return true;
        }
        if(root==nullptr || subRoot==nullptr || root->val!=subRoot->val) {
            return false;
        }
        return root->val==subRoot->val && sametree(root->left, subRoot->left) && 
            sametree(root->right, subRoot->right);
    }
};

// 两数之和
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, vector<int> > num_index;
        for(int i=0; i<nums.size(); i++) {
            num_index[nums[i]].push_back(i);
        }
        for (int i=0; i<nums.size(); i++) {
            int num = nums[i];
            int diff= target-num;
            if(num_index.find(diff)!=num_index.end()){
                vector<int> idxs = num_index[diff];
                auto it = std::find(idxs.begin(), idxs.end(), i);
                if (it != idxs.end()) {
                    idxs.erase(it);
                }
                if (!idxs.empty()) {
                    return {i, idxs[0]};
                }
            }
        }
        return {};
    }
};

// 反转链表
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur!=nullptr){
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
};

// 合并有序数组
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int inx = m+n-1; int i=m-1; int j=n-1;
        while (i>=0 && j>=0) {
            if (nums1[i]>nums2[j]) {
                nums1[inx] = nums1[i];
                i--;
            } else {
                nums1[inx] = nums2[j];
                j--;
            }
            inx--;
        }
        while (j>=0) {
            nums1[inx]=nums2[j];
            inx--;
            j--;
        }
    }
};

// 乘积最大子数组
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int n = nums.size();
        vector<int> min(nums);
        vector<int> max(nums);
        for(int i=1; i<n; i++) {
            min[i] = std::min(min[i-1]*nums[i], std::min(nums[i], max[i-1]*nums[i]));
            max[i] = std::max(max[i-1]*nums[i], std::max(nums[i], min[i-1]*nums[i]));
        }
        vector<int>::iterator ans=max_element(max.begin(), max.end());
        return *ans;
    }
};

// 二叉搜索树转换为双向链表
class Solution {
public:
    Node* treeToDoublyList(Node* root) {
        if (root==nullptr) {
            return root;
        }
        Node* head = nullptr;
        Node* pre = nullptr;
        inorder(root, head, pre);
        if (head) {
            head->left = pre;
            pre->right = head;
        }
        return head;
    }
private:
    void inorder(Node* root, Node*& head, Node*& pre) {  // 这里需要传递指针的引用
        if (root==nullptr) {
            return ;
        }
        inorder(root->left, head, pre);
        if (head==nullptr) {
            head = root;
        } else {
            pre->right = root;
            root->left = pre;
        }
        pre = root;
        inorder(root->right, head, pre);
    }
};

// 组合最大数
class Solution {
public:
    string largestNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end(), 
             [](int x, int y) {
             string a = to_string(x);
             string b = to_string(y);
             return a+b > b+a;
             });
        if (nums[0]==0) {
            return "0";
        }
        string result;
        for(int num: nums) {
            result += to_string(num);
        }
        return result;
    }
};

// 逆序对总数
// 1. 使用归并排序
class Solution {
public:
    int mergeSort(vector<int>& record, vector<int>& tmp, int l, int r) {
        //归并排序函数，建了两个容器，整数数组record临时数组tmp左边界l右边界r
        if (l >= r) {//左数大于右数
            return 0;
        }

        int mid = (l + r) / 2;//把一列数分成左右两部分
        int inv_count = mergeSort(record, tmp, l, mid) + mergeSort(record, tmp, mid + 1, r);//递归
        int i = l, j = mid + 1, pos = l;//i指向左半部分的起始位置，j指向右半部分的起始位置，pos指向临时数组的起始位置
        while (i <= mid && j <= r) {
            if (record[i] <= record[j]) {
                tmp[pos] = record[i];
                ++i;
                inv_count += (j - (mid + 1));
            }
            else {
                tmp[pos] = record[j];
                ++j;
            }
            ++pos;
        }
        for (int k = i; k <= mid; ++k) {//左
            tmp[pos++] = record[k];
            inv_count += (j - (mid + 1));//记录逆序对
        }
        for (int k = j; k <= r; ++k) {//右
            tmp[pos++] = record[k];
        }
        copy(tmp.begin() + l, tmp.begin() + r + 1, record.begin() + l);////将排好序的临时数组复制回原数组
        return inv_count;//逆序对的数量
    }

    int reversePairs(vector<int>& record) {
        int n = record.size();
        vector<int> tmp(n);
        return mergeSort(record, tmp, 0, n - 1);
    }
};

// 使用树状数组
class Solution {
public:
    int reversePairs(vector<int>& record) {
        if(record.size()<2)
        {
            return 0;
        }
        //树状数组/线段树
        int ans=0;
        int n=record.size();
        vector<int> trees(n+1);

        auto lowerBit=[&](int x)->int
        {
            return x&(-x);
        };

        auto add=[&](int x,int u)
        {
            for(int i=x;i<=n;i+=lowerBit(i))
            {
                trees[i]+=u;
            }
        };

        auto query=[&](int x)
        {
            int ans=0;
            for(int i=x;i;i-=lowerBit(i))
            {
                ans+=trees[i];
            }
            return ans;
        };

        //从大到小插入
        int id[n];
        iota(id,id+n,0);
        sort(id,id+n,[&](int i,int j){return record[i]>record[j];});
        int pre=record[id[0]];
        vector<int> tmp;
        for(int i:id)
        {
            if(record[i]!=pre)
            {
                for(int j:tmp)
                {
                    add(j+1,1);
                }
                vector<int> tmp1;
                swap(tmp,tmp1);
                pre=record[i];
            }
            tmp.push_back(i);
            ans+=query(i);

        }
        return ans;
    }
};

class BIT {
private:
    vector<int> tree;
    int n;

public:
    BIT(int _n): n(_n), tree(_n + 1) {}

    static int lowbit(int x) {
        return x & (-x);
    }

    int query(int x) {
        int ret = 0;
        while (x) {
            ret += tree[x];
            x -= lowbit(x);
        }
        return ret;
    }

    void update(int x) {
        while (x <= n) {
            ++tree[x];
            x += lowbit(x);
        }
    }
};

class Solution {
public:
    int reversePairs(vector<int>& record) {
        int n = record.size();
        vector<int> tmp = record;
        // 离散化
        sort(tmp.begin(), tmp.end());
        for (int& num: record) {
            num = lower_bound(tmp.begin(), tmp.end(), num) - tmp.begin() + 1;
        }
        // 树状数组统计逆序对
        BIT bit(n);
        int ans = 0;
        for (int i = n - 1; i >= 0; --i) {
            ans += bit.query(record[i] - 1);
            bit.update(record[i]);
        }
        return ans;
    }
};

// 重排链表
class Solution {
public:
    void reorderList(ListNode* head) {
        if (!head || !head->next || !head->next->next) {
            return;
        }
        ListNode* slow = head;
        ListNode* fast = head;
        while(fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode* _l2 = slow->next;
        slow->next = nullptr;
        ListNode* l1 = head;
        ListNode* l2 = reverse(_l2); 
        merge(l1, l2);
    }
private:
    ListNode* reverse(ListNode* head) {
        ListNode* cur = head;
        ListNode* pre = nullptr;
        ListNode* next;
        while (cur!=nullptr) {
            next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
    void merge(ListNode* l1, ListNode* l2) {
        ListNode *l1next, *l2next;
        while (l1!=nullptr && l2!=nullptr) {
            l1next = l1->next;
            l2next = l2->next;

            l1->next = l2;
            l1 = l1next;

            l2->next = l1;
            l2 = l2next;
        }
    }
};

// 除自身以外数组的乘积
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int size = nums.size();
        vector<int> left(size, 1);
        vector<int> right(size, 1);
        for (int i=1; i<size; i++) {
            left[i] = left[i-1]*nums[i-1];
            right[size-i-1] = nums[size-i] * right[size-i];
        }
        vector<int> ans ;
        for (int i=0; i<size; i++) {
            ans.push_back(left[i]*right[i]);
        }
        return ans;
    }
};

// k 个一组翻转
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode dummynode = ListNode(-1);
        dummynode.next = head;

        ListNode* pre = &dummynode;
        while (head) {
            ListNode *tail = pre;
            for (int i=0; i<k; i++) {
                tail = tail->next;
                if (!tail) {
                    return dummynode.next;
                }
            }
            ListNode* next = tail->next;
            reverse(head, tail);
            pre->next = head;
            tail->next = next;
            pre = tail;
            head = next;
        }
        return dummynode.next;
    }

private:
    void reverse(ListNode*& head, ListNode*& tail) {
        ListNode* pre = head;
        ListNode* cur = head;
        while (pre!=tail) {
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        std::swap(head, tail);
    }
};

// 求根节点到叶节点数字之和
class Solution {
public:
    int sumNumbers(TreeNode* root) {
        sum = 0;
        dfs(root, 0);
        return sum;
    }
private:
    int sum;
    void dfs(TreeNode* root, int target) {
        if (!root->left && !root->right) {
            sum += (target*10 + root->val);
            return ;
        }

        if (root->left) {
            dfs(root->left, target*10 + root->val);
        }
        if (root->right) {
            dfs(root->right, target*10 + root->val);
        }
    }
};

// 接雨水
class Solution {
public:
    int trap(vector<int>& height) {
        int size = height.size();
        vector<int> left_max(size, height[0]);
        vector<int> right_max(size, height[size-1]);

        for(int i=1; i<size; i++){
            left_max[i] = std::max(left_max[i-1], height[i-1]);
        }
        for(int i=size-2; i>-1; i--) {
            right_max[i]= std::max(right_max[i+1], height[i+1]);
        }

        int ans;
        for(int i=1; i<size-1; i++) {
            int minmax = std::min(left_max[i], right_max[i]);
            ans += (minmax - height[i]);
        }
        return ans;
    }
};

// 字符串相加
//
class Solution {
public:
    string addStrings(string num1, string num2) {
        int l1 = num1.size();
        int l2 = num2.size();
        int ctx=0;
        string ans="";
        while (l1>=0 || l2>=0 || ctx) {
            /* int a=0; */
            /* int b=0; */
            /* if (l1>=0) { */
            /*     char c1 = num1[l1]; */
            /*     l1--; */
            /*     a = (c1-'0') */
            /* } */
            /* if(l2>=0) { */
            /*     c2 = num2[l2]; */
            /*     l2--; */
            /*     b = (c2-'0') */
            /* } */
            int a = ( l1>=0 ? num1[l1]-'0' : 0 );
            int b = ( l2>=0 ? num2[l2]-'0' : 0 );
            int tmp = a + b + ctx;
            int x = tmp % 10;
            ans.push_back('0'+x)
            ctx = tmp/10;
            l1--;
            l2--;
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
};

// 最大正方形
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.size()==0 || matrix[0].size()==0) {
            return 0;
        }
        int m = matrix.size();
        int n = matrix[0].size();
        vector< vector<int> > dp(m, vector<int>(n, 0));

        int maxside = 0;
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                if (matrix[i][j] == '1') {
                    if (i==0 || j==0) {
                        dp[i][j] = 0;
                    } else {
                        dp[i][j] = min(dp[i-1][j-1], min(dp[i][j-1], dp[i-1][j])) + 1;
                    }
                }
            }
        }
        return maxside * maxside;
    }
};

// 第N位数字
class Solution {
public:
    int findNthDigit(int n) {
        long long d=1; // 表示当前数字有几位;
        long long count=9;
        while (n>d*count) {
            n -= d*count;
            d++;
            count *= 10;
        }
        long long index = n-1;
        long long start = pow(10, d-1);
        long num = start + index/d;
        int _ix = index % d;
        return (to_string(num)[_ix]-'0');
    }
};

// 螺旋矩阵 
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        int total = m*n;
        int left=0, right=n-1, top=0, bottom=m-1;
        vector<int> ans;
        while (total>=0) {
            if (left > right) break;
            for(int i=left; i<=right; i++) {
                ans.push_back(matrix[top][i]);
            }
            top += 1;

            if ( top > bottom ) break;
            for(int i=top; i<=bottom; i++) {
                ans.push_back(matrix[i][right]);
            }
            right--;

            if ( right < left) break;
            for(int i=right; i>=left; i--) {
                ans.push_back(matrix[bottom][i]);
            }
            bottom--;

            if (bottom < top) break;
            for(int i=bottom; i>=top; i--) {
                ans.push_back(matrix[i][left]);
            }
            left++;
        }
        return ans;
    }
};

// 二叉树最大深度
class Solution {
public:
    int maxDepth(TreeNode* root) {
        return dfs(root);
    }
private:
    int dfs(TreeNode* root) {
        if (!root) {
            return 0;
        }
        return std::max(dfs(root->left), dfs(root->right)) + 1;
    }
};

// 相交链表
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA || !headB) {
            return nullptr;
        }
        ListNode* p = headA;
        ListNode* q = headB;
        while(p!=q) {
            p = p ? p->next : headB; 
            q = q ? q->next : headA;
        }
        return p;
    }
};

// 二叉树剪枝
class Solution {
public:
    TreeNode* pruneTree(TreeNode* root) {
        if (!root) {
            return nullptr;
        }
        root->left = pruneTree(root->left);
        root->right= pruneTree(root->right);
        if (!root->left && !root->right && root->val==0) {
            return nullptr;
        }
        return root;
    }
};

// 二叉树序列化和反序列化
class Codec {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        vector<string> ans;
        if (!root) {
            return "[]";
        }
        deque< TreeNode* > dq = {root};
        while(!dq.empty()) {
            TreeNode* node = dq.front();
            dq.pop_left();
            if (node) {
                ans.push_back(to_string(node->val));
                dq.push_back(node->left);
                dq.push_back(node->right);
            } else {
                ans.push_back("None");
            }
        }
        string serial = "";
        for (string t : ans) {
            serial += ( "," + t );
        }
        return "[" + serial + "]";
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if (data == "[]") {
            return nullptr;
        }
        std::vector<std::string> datalist;
        size_t start = 1;
        while (start < data.size()) {
            size_t end = data.find(',', start);
            datalist.push_back(data.substr(start, end - start));
            start = end + 1;
        }
        TreeNode* root = new TreeNode(std::stoi(datalist[0]));
        std::queue<TreeNode*> q;
        q.push(root);
        size_t i = 1;
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            if (datalist[i] != "None") {
                node->left = new TreeNode(std::stoi(datalist[i]));
                q.push(node->left);
            }
            ++i;
            if (datalist[i] != "None") {
                node->right = new TreeNode(std::stoi(datalist[i]));
                q.push(node->right);
            }
            ++i;
        }
        return root;
    }
};

// 两数相加
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        l1 = reverse(l1);
        l2 = reverse(l2);

        int ctx = 0;
        ListNode* pre = nullptr;
        while(l1 || l2 || ctx) {
            int a = l1 ? l1->val : 0;
            int b = l2 ? l2->val : 0;
            int sum = a+b+ctx;
            int val = sum % 10;
            ctx = sum / 10;
            ListNode* cur = new ListNode(val);
            cur->next = pre;
            pre = cur;
            if (l1) {
                l1 = l1->next;
            }
            if (l2) {
                l2 = l2->next;
            }
        }
        return pre;
    }
private:
    ListNode* reverse(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur) {
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
};

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int ctx = 0;
        ListNode* dummynode = new ListNode(-1);
        ListNode* pre = dummynode;
        while (l1 || l2 || ctx) {
            int a = l1 ? l1->val : 0;
            int b = l2 ? l2->val : 0;
            int sum = (a+b+ctx);
            int val = sum % 10;
            ctx = sum / 10;
            ListNode* cur = new ListNode(val);
            pre->next = cur;
            pre = cur;
            if (l1) {
                l1 = l1->next;
            }
            if (l2) {
                l2 = l2->next;
            }
        }
        return dummynode->next;
    }
};

// 不同的路径
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector< vector<int> > dp(m, vector<int>(n, 0));
        for (int i=0; i<m; i++){
            dp[i][0] = 1;
        }
        for (int j=0; j<n; j++){
            dp[0][j] = 1;
        }
        
        for(int i=1; i<m; i++) {
            for (int j=1; j<n; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};

class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();

        vector< vector<int> > dp(m, vector<int>(n, 0));
        dp[0][0] = (obstacleGrid[0][0]==1) ? 0 : 1;

        for(int i=1; i<m; ++i) {
            dp[i][0] = (obstacleGrid[i][0]==1) ? 0 : dp[i-1][0];
        }

        for (int j=1; j<n; ++j) {
            dp[0][j] = (obstacleGrid[0][j]==1) ? 0 : dp[0][j-1];
        }

        for(int i=1; i<m; i++) {
            for (int j=1; j<n; j++) {
                if (obstacleGrid[i][j]==1) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }

        return dp[m-1][n-1];
    }
};

// 最长连续序列
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> num_set;
        for (auto num: nums) {
            num_set.insert(num);
        }
        int maxlen = 0;
        for (int num: nums) {
            if (num_set.count(num-1)) {
                continue;
            }

            int start = num;
            int end = num+1;
            while (num_set.count(end)){
                end = end+1;
            }
            maxlen = std::max(maxlen, end-start);
        }
        return maxlen;
    }
};

// 丑数
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> dp(n+1);
        dp[0] = 1;
        dp[1] = 1;
        int k2=1, k3=1, k5=1;
        for(int i=2; i<n+1; i++) {
            dp[i] = std::min(2*dp[k2], std::min(3*dp[k3], 5*dp[k5]));
            if (dp[i]==2*dp[k2]) {
                k2++;
            } 
            if (dp[i]==3*dp[k3]) {
                k3++;
            }
            if (dp[i]==5*dp[k5]) {
                k5++;
            }
        }
        return dp[n];
    }
};

// 路径总和 III
class Solution {
public:
    int pathSum(TreeNode* root, int targetSum) {
        prefixsumCount.clear();
        prefixsumCount[0] = 1;
        int cnt = dfs(root, 0, targetSum);
        return cnt;
    }
private:
    unordered_map<long, int> prefixsumCount;
    int dfs(TreeNode* root, long prefixsum, int target) {
        if (!root) {
            return 0;
        }
        int cnt = 0;
        prefixsum += root->val;
        if (prefixsumCount.find(prefixsum-target)!=prefixsumCount.end()) {
            cnt += prefixsumCount[prefixsum-target];
        }
        if (prefixsumCount.find(prefixsum)!=prefixsumCount.end()) {
            prefixsumCount[prefixsum]++;
        } else {
            prefixsumCount[prefixsum] = 1;
        }
        cnt += dfs(root->left, prefixsum, target);
        cnt += dfs(root->right, prefixsum, target);
        prefixsumCount[prefixsum]--;

        return cnt;
    }
};

// 和为k的子数组
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<long, int> prefixcount;
        prefixcount[0] = 1;
        long prefix = 0;
        int ans = 0;
        for (int num: nums) {
            prefix += num;
            if (prefixcount.find(prefix-k)!=prefixcount.end()) {
                ans += prefixcount[prefix-k];
            }
            if (prefixcount.find(prefix)!=prefixcount.end()) {
                prefixcount[prefix]++;
            } else {
                prefixcount[prefix] = 1;
            }
        }
        return ans;
    }
};

// 多数元素
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int cur=nums[0], count=1;
        // unordered_map<int, int> map;
        int size = nums.size();
        for (int i=1; i<size; i++) {
            if (count==0) {
                cur = nums[i];
                count = 1;
                continue;
            }

            if (nums[i]==cur) {
                count++;
            } else {
                count--;
            }
        }
        return cur;
    }
};

// 环形链表II 
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if(!head || !head->next) {
            return nullptr;
        }
        ListNode* slow = head;
        ListNode* fast = head;
        while(true) {
            if (!fast->next || !fast->next->next) {
                return nullptr;
            }
            slow = slow->next;
            fast = fast->next->next;
        }
        fast = head;
        while(fast!=slow) {
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }
};

// 最长公共子序列
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size();
        int n = text2.size();
        vector< vector<int> > dp(m+1, vector<int>(n+1, 0));
        for (int i=1; i<m+1; i++) {
            for (int j=1; j<n+1; j++) {
                if (text1[i-1]==text2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[m][n];
    }
};

// 二叉树最大路径和
class Solution {
public:
    int maxPathSum(TreeNode* root) {
        maxsum = LONG_MIN;
        long t = maxroot(root);
        return maxsum;
    }

private:
    long maxsum;
    long maxroot(TreeNode* root) {
        if (!root) {
            return 0;
        }
        long left = maxroot(root->left);
        long right= maxroot(root->right);
        long sum = left + right + root->val;
        if (sum > maxsum) {
            maxsum=sum;
        }
        long _max = std::max(left, right)+root->val;
        return _max > 0 ? _max : 0 ;
    }
};

// 验证前序遍历序列二叉搜索树
class Solution {
public:
    bool verifyPreorder(vector<int>& preorder) {
        if(preorder.size() <= 2) return true;
        int MIN = INT32_MIN;
        stack<int> s;
        for(int i = 0; i < preorder.size(); ++i)
        {
            if(preorder[i] < MIN)
                return false;
            while(!s.empty() && s.top() < preorder[i])//遇到大的了，右分支
            {
                MIN = s.top();//记录弹栈的栈顶为最小值
                s.pop();
            }
            s.push(preorder[i]);
        }
        return true;
    }
};

// 二叉搜索树的第k大节点
class Solution {
public:
    int findTargetNode(TreeNode* root, int cnt) {
        this->cnt = cnt;
        dfs(root);
        return ans
    }
private:
    int ans, cnt;
    void dfs(TreeNode* root){
        if (!root) {
            return ;
        }
        dfs(root->right);
        if (cnt==0) {
            return ;
        }
        if (--cnt==0) {
            ans = root->val;
        }
        dfs(root->left);
    }
};

// 最长重复子数组
//
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        vector< vector<int> > dp(m+1, vector<int>(n+1, 0));

        int max = LONG_MIN;
        for (int i=1; i<m+1; i++) {
            for(int j=1; j<n+1; j++) {
                if (nums1[i-1]!=nums2[j-1]) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }
                max = std::max(max, dp[i][j]);
            }
        }
        return max;
    }
};

// 最长上升子序列个数
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 0);
        vector<int> cnt(n, 1);
        int ans=0;
        int maxlen = INT_MIN;
        for (int i=0; i<n; i++) {
            for (int j=0; j<i; j++) {
                if (nums[i]>nums[j]) {
                    if (dp[i]<dp[j]+1) {
                        dp[i] = dp[j]+1;
                        cnt[i] = cnt[j];
                    } else if (dp[i]==dp[j]+1) {
                        cnt[i] += cnt[j];
                    }
                }
            }
            if (maxlen < dp[i]) {
                maxlen = dp[i];
                ans  = cnt[i];
            } else if (maxlen==dp[i]) {
                ans += cnt[i];
            }
        }
        return ans;
    }
};

// 字典序的第k小数字
class Solution {
public:
    int findKthNumber(int n, int k) {
        long cur=1;
        k = k-1;
        while (k) {
            long steps = getSteps(n, cur);
            if (steps <= k) {
                k -= steps;
                cur += 1;
            } else {
                cur *= 10;
                k -= 1;
            }
        }
        return cur;
    }
private:
    int getSteps(long n, long cur){
        long steps=0;
        long first=cur, last=cur;
        while (first<=n) {
            steps += std::min(n, last)-first+1;
            first *= 10;
            last = last*10 + 9;
        }
        return steps;
    }
};

// 递增三元组子序列
class Solution {
public:
    bool increasingTriplet(vector<int>& nums) {
        int first = nums[0];
        int second = INT_MAX;
        for (int i=1; i<nums.size(); i++) {
            if (nums[i]>second) {
                return true;
            } else {
                if (nums[i]<=first) {
                    first = nums[i];
                } else {
                    second = std::min(nums[i], second);
                }
            }
        }
        return false;
    }
};

// rand7 -> rand10
class Solution {
public:
    int rand10() {
        while (true) {
            int a = rand7();
            int b = rand7();
            int x = (a-1)*b + 7;
            if (x<=40) {
                return 1 + x%10;
            }
        }
    }
}

// 找到k个最接近的元素
class Solution {
public:
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        int n = arr.size();
        if (n<=k) {
            return arr;
        }
        int right = std::lower_bound(arr.begin(), arr.end(), x) - arr.begin();
        int left = right-1;
        for (int i=0; i<k; i++) {
            if(left<0) {
                right++;
            } else if (right>=n) {
                left--;
            } else if (x-arr[left]<=arr[right]-x) {
                left--;
            } else {
                right++;
            }
        }
        return vector<int>(arr.begin()+left+1, arr.begin()+right);
    }
};

// 不同的二叉搜索树
class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(n+1, 1);
        dp[0]=1;
        dp[1]=1;
        for (int i=2; i<n+1; i++) {
            for (int j=1; j<i+1; j++){
                dp[i] += dp[j-1]*dp[i-j];
            }
        }
        return dp[n];
    }
};

// 缺失的第一个正数
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int i=0; i<n; i++) {
            while (nums[i]>=1 && nums[i]<=n && (nums[i]!=nums[nums[i]-1])) {
                std::swap(nums[i], nums[nums[i]-1]);
            }
        }
        for (int i=0; i<n; i++) {
            if (nums[i]!=(i+1)) {
                return i+1;
            }
        }
        return n+1;
    }
};

// 买卖股票的最佳时机III
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int buy1=-prices[0], buy2=-prices[0];
        int sell1=0, sell2=0;
        for (int i=1; i<n; i++){
            buy1 = std::max(buy1, -prices[i]);
            sell1 = std::max(buy1+prices[i], sell1);
            buy2 = std::max(buy2, sell1-prices[i]);
            sell2= std::max(buy2+prices[i], sell2);
        }
        return sell2;
    }
};

// 二叉树右视图 
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        if (!root) {
            return {};
        }
        std::queue<TreeNode*> stack;
        stack.push(root);
        vector<int> ans;
        while(!stack.empty()) {
            int size = stack.size();
            std::vector<int> tmp;
            for(int i=0; i<size; i++) {
                TreeNode* node = stack.front();
                stack.pop();
                tmp.push_back(node->val);
                if (node->left) {
                    stack.push(node->left);
                }
                if (node->right) {
                    stack.push(node->right);
                }
            }
            ans.push_back(tmp.back());
        }
        return ans;

    }
};

// 搜索二维矩阵
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        int n = matrix[0].size();

        int i=0;
        int j=n-1;
        while(i<m && i>=0 && j<n && j>=0) {
            if (matrix[i][j]==target) {
                return true;
            } else if (matrix[i][j]<target) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }
};

//矩阵中最长递增子序列
class Solution {
public:
    static constexpr int dirs[4][2] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int rows, cols;
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        if (matrix.size()==0 || matrix[0].size()==0) {
            return 0;
        }
        rows = matrix.size();
        cols = matrix[0].size();
        auto memo = vector< vector<int> > (rows, vector<int> (cols));
        int ans = 0;
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                ans = std::max(ans, dfs(matrix, i, j, memo));
            }
        }
        return ans;
    }
    int dfs(vector< vector<int> > &matrix, int row, int col, vector< vector<int> > &memo) {
        if (memo[row][col]!=0) {
            return memo[row][col];
        }
        ++memo[row][col];
        for (int i=0; i<4; ++i){
            int nrow = row+dirs[i][0], ncol = col+dirs[i][1];
            if(nrow>=0 && nrow<rows && ncol>=0 && ncol<cols && matrix[nrow][ncol]>matrix[row][col]) {
                memo[row][col] = max(memo[row][col], dfs(matrix, nrow, ncol, memo)+1);
            }
        }
        return memo[row][col];
    }
};

// 寻找旋转排序数组中的最小值
class Solution {
public:
    int findMin(vector<int>& nums) {
        int n = nums.size();
        int left=0, right=n-1;
        while (left<right) {
            int mid = (left+right)/2;
            if (nums[mid] > nums[right]) {
                left = mid+1;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }
};

// 圆圈中最后剩下的数字
class Solution {
public:
    int lastEle(int num, int target) {
        int pos = 0;
        for(int i=2; i<num+1; i++) {
            pos = (pos + target) % i;
        }
        return pos;
    }
};

// 划分为k个相等的子集
class Solution {
public:
    bool canPartionKSubsets(vector<int> &nums, int k) {
        int total = accumulate(nums.begin(), nums.end(), 0);
        int mod = total % k;
        if(mod) {
            return false;
        }
        int avg = total / k;
        std::sort(nums.begin(), nums.end());
        vector<int> cur(k, 0);
        return dfs(0, nums, cur);
    }
private:
    bool dfs(int i, vector<int>& nums, vector<int>& cur, int k, int avg) {
        if (i==nums.size()) {
            return true;
        }
        for (int j=0; j<k; j++) {
            if(j>0 && cur[j]==cur[j-1]) {
                continue;
            } 
            cur[j] += nums[i];
            if (cur[j]<=avg && dfs(i+1, nums, cur, k, avg)) {
                return true
            }
            cur[j] -= nums[i];
        }
        return false;
    }
};

// 下一个排列
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        // 从后往前找，找到第一个降序位置，交换，然后将后面的重排；
        int n = nums.size();
        int i=n-2, j = n-1;
        while (i>=0) {
            if (nums[i]<nums[j]) {
                break;
            }
            i -= 1;
            j -= 1;
        }
        if (i<0) {
            return std::reverse(nums.begin(), nums.end());
        }
        int k=n-1;
        while (k>=j) {
            if (nums[k]<=nums[i]) {
                k--;
            } else {
                std::swap(nums[k], nums[i]);
                break;
            }
        }
        std::reverse(nums.begin()+j, nums.end());
    }
};
