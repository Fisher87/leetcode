/*================================================================
*   Copyright (C) 2023 Fisher. All rights reserved.
*   
*   文件名称：algorithm.cpp
*   创 建 者：YuLianghua
*   创建日期：2023年08月01日
*   描    述：
*
================================================================*/
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
            if (!dq.empyt() && i-dq.front()>=k) {
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
