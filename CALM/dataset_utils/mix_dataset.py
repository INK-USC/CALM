import os
from pathlib import Path

def getMergedContent(fps1, fpt1, fps2, fpt2, fps3=None, fpt3=None):
    """
    fps1 = file pointer of source 1
    fpt1 = file pointer of target 1 and similarly for other sources
    """
    merged_content_source = ""
    merged_content_target = ""

    break_flag = 0
    while break_flag != 1:
        s1_flag = s2_flag = s3_flag = 0

        # Adding one line from source 1
        fps1_line = fps1.readline()
        fpt1_line = fpt1.readline()

        if len(fps1_line) != 0:
            merged_content_source += fps1_line
            merged_content_target += fpt1_line
        else:
            # Done reading the source 1
            s1_flag = 1

        # Adding one line from source 2
        fps2_line = fps2.readline()
        fpt2_line = fpt2.readline()

        if len(fps2_line) != 0:
            merged_content_source += fps2_line
            merged_content_target += fpt2_line
        else:
            # Done reading the source 2
            s2_flag = 1

        # Adding one line from source 3 if it exists (optional; to support merging of two (or) three files)
        if fps3 is not None:
            fps3_line = fps3.readline()
            fpt3_line = fpt3.readline()

            if len(fps3_line) != 0:
                merged_content_source += fps3_line
                merged_content_target += fpt3_line
            else:
                # Done reading the source 3
                s3_flag = 1
        else:
            s3_flag = 1

        if fps3 is not None:
            if s1_flag == 1 or s2_flag == 1 or s3_flag == 1:
                # If done reading any one of the files, break from the loop
                break_flag = 1
        else:
            if s1_flag == 1 or s2_flag == 1:
                # If done reading any one of the files, break from the loop
                break_flag = 1

    return merged_content_source, merged_content_target


if __name__ == "__main__":
    concept_fp = open('datasets/cor/train.source')
    keyword_fp = open('datasets/c2s/train.source')
    option2_fp = open('datasets/option2/train.source')

    concept_fp_t = open('datasets/cor/train.target')
    keyword_fp_t = open('datasets/c2s/train.target')
    option2_fp_t = open('datasets/option2/train.target')

    valid_concept_fp = open('datasets/cor/dev.source')
    valid_keyword_fp = open('datasets/c2s/dev.source')
    valid_option2_fp = open('datasets/option2/dev.source')

    valid_concept_fp_t = open('datasets/cor/dev.target')
    valid_keyword_fp_t = open('datasets/c2s/dev.target')
    valid_option2_fp_t = open('datasets/option2/dev.target')

    # For the mixup dataset:
    merged_source_train1, merged_target_train1 = getMergedContent(concept_fp, concept_fp_t, keyword_fp, keyword_fp_t, option2_fp, option2_fp_t)
    merged_source_valid1, merged_target_valid1 = getMergedContent(valid_concept_fp, valid_concept_fp_t, valid_keyword_fp, valid_keyword_fp_t, valid_option2_fp, valid_option2_fp_t)

    output_dir = "datasets/mix"
    output_dir1 = Path(output_dir)
    output_dir1.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir1, "train.source"), "w") as f:
        f.write(merged_source_train1)
    with open(os.path.join(output_dir1, "train.target"), "w") as f:
        f.write(merged_target_train1)
    with open(os.path.join(output_dir1, "dev.source"), "w") as f:
        f.write(merged_source_valid1)
    with open(os.path.join(output_dir1, "dev.target"), "w") as f:
        f.write(merged_target_valid1)




