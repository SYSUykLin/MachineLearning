class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        arr = nums
        nums = sorted(nums)
        index = self.get_large_index(nums , target)
        for i in range(index):
            for j in range(i+1 , index):
                if nums[i]+nums[j] == target:
                    if arr.index(nums[i]) == arr.index(nums[j]):
                        return [arr.index(nums[i]) , arr.index(nums[j] , arr.index(nums[i])+1 , len(nums))]
                    else:
                        if arr.index(nums[i]) > arr.index(nums[j]):
                            return [arr.index(nums[j]),arr.index(nums[i])]
                        return  [arr.index(nums[i]),arr.index(nums[j])]
    def get_large_index(self,nums , target):
        an = len(nums)
        for i , n in enumerate(nums):
            if n >= target:
                an = i
        return an
        pass


