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
