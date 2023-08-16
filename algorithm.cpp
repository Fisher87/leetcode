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
