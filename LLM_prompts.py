class GetPrompts():
    def __init__(self):
        self.prompt_task = "You are requested to design a novel algorithm that, given a set of augmentation techniques, selects several of them based on the change in per-class accuracy between the current training time point and the previous one, to be employed in the subsequent training phase, with the aim of enhancing the model's ability to tackle long-tail problems. This algorithm should deviate from existing methodologies present in the literature."
        self.prompt_func_name = "get_aug_type"
        self.prompt_func_inputs = ["aug_weight","ACCs","History_ACCs","lats_chose_matix","lats_chose_exts","epoch"]
        self.prompt_func_outputs = ["chose_matrix,chose_exts"]
        self.prompt_inout_inf = '''aug_weight is a two-dimensional integer array initialized to 1, used to record the historical weight information for each category (indexed by rows) across every augmentation technique (indexed by columns). ACCs is a one-dimensional integer array showcasing the performance of each category at the current training instant, specifically, the count of correct predictions within that category. History_ACCs is a two-dimensional integer array that records the number of correct predictions made by each augmentation technique (column-wise) the last time they were employed for every category. lats_chose_matrix is a two-dimensional Boolean array indicating whether specific augmentation techniques (by column index) were utilized for each category (row index) in the previous training step; True signifies usage, while False denotes non-usage. lats_chose_exts is a two-dimensional float array representing the application intensity of each augmentation method across classifications at the current point in time, with a range from 0 to 1, where higher numbers imply greater enhancement strength. epoch is an integer denoting the current training epoch, indicating the number of completed training cycles. chose_matrix is a two-dimensional Boolean array marking which augmentation techniques (by column index) will be employed in the next training step; True values indicate adoption, and False, rejection. chose_exts is a two-dimensional float array signifying the intensity of applying each augmentation technique to individual categories in the upcoming time step, also ranging from 0 to 1, with larger values indicating more substantial augmentation efforts.'''
        self.prompt_other_inf = "All are Numpy arrays."

    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf


if __name__ == "__main__":
    getprompts = GetPrompts()
    print(getprompts.get_task())
