import os

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
    concept_fp = open('../datasets/cor/train.source')
    keyword_fp = open('../datasets/c2s/train.source')
    option2_fp = open('../datasets/option2/train.source')

    concept_fp_t = open('../datasets/cor/train.target')
    keyword_fp_t = open('../datasets/c2s/train.target')
    option2_fp_t = open('../datasets/option2/train.target')

    valid_concept_fp = open('../datasets/cor/valid.source')
    valid_keyword_fp = open('../datasets/c2s/valid.source')
    valid_option2_fp = open('../datasets/option2/valid.source')

    valid_concept_fp_t = open('../datasets/cor/valid.target')
    valid_keyword_fp_t = open('../datasets/c2s/valid.target')
    valid_option2_fp_t = open('../datasets/option2/valid.target')

    # For the mixup dataset:
    merged_source_train1, merged_target_train1 = getMergedContent(concept_fp, concept_fp_t, keyword_fp, keyword_fp_t, option2_fp, option2_fp_t)
    merged_source_valid1, merged_target_valid1 = getMergedContent(valid_concept_fp, valid_concept_fp_t, valid_keyword_fp, valid_keyword_fp_t, valid_option2_fp, valid_option2_fp_t)

    output_dir1 = "datasets/mix"
    with open(os.path.join(output_dir1, "train.source"), "w") as f:
        f.write(merged_source_train1)
    with open(os.path.join(output_dir1, "train.target"), "w") as f:
        f.write(merged_target_train1)
    with open(os.path.join(output_dir1, "valid.source"), "w") as f:
        f.write(merged_source_valid1)
    with open(os.path.join(output_dir1, "valid.target"), "w") as f:
        f.write(merged_target_valid1)

    # # Reset the file pointer to read the file again
    # concept_fp.seek(0)
    # keyword_fp.seek(0)
    # concept_fp_t.seek(0)
    # keyword_fp_t.seek(0)
    #
    # valid_concept_fp.seek(0)
    # valid_keyword_fp.seek(0)
    # valid_concept_fp_t.seek(0)
    # valid_keyword_fp_t.seek(0)
    #
    # # For the generator only dataset:
    # merged_source_train2, merged_target_train2 = getMergedContent(concept_fp, concept_fp_t, keyword_fp, keyword_fp_t)
    # merged_source_valid2, merged_target_valid2 = getMergedContent(valid_concept_fp, valid_concept_fp_t, valid_keyword_fp, valid_keyword_fp_t)
    #
    # output_dir2 = "datasets/mix"
    # with open(os.path.join(output_dir2, "train.source"), "w") as f:
    #     f.write(merged_source_train2)
    # with open(os.path.join(output_dir2, "train.target"), "w") as f:
    #     f.write(merged_target_train2)
    # with open(os.path.join(output_dir2, "valid.source"), "w") as f:
    #     f.write(merged_source_valid2)
    # with open(os.path.join(output_dir2, "valid.target"), "w") as f:
    #     f.write(merged_target_valid2)




