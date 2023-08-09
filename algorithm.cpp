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
