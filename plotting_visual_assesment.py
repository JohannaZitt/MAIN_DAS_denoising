import os


experiments = os.listdir("experiments")

for experiment in experiments:

    ablation_path = os.path.join("experiments", experiment, "plots", "ablation", "0706_RA88")
    accumulation_path = os.path.join("experiments", experiment, "plots", "accumulation", "0706_AJP")

    categories = ["1_raw:visible_denoised:better_visible",
                  "2_raw:not_visible_denoised:visible",
                  "3_raw:not_visible:denoised:not_visible"]

    category_sum = 0

    for category in categories:

        category_size = len(os.listdir(os.path.join(ablation_path, category)))
        category_sum += category_size
        print(category_size)

    print("SUM: ", experiment, category_sum)
