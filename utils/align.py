import numpy as np
def global_align(gtj0, prj0, key):
    gtj = gtj0.copy()
    prj = prj0.copy()

    if key in ["stb", "rhd"]:
        # gtj :B*21*3
        # prj :B*21*3
        root_idx = 9  # root
        ref_bone_link = [0, 9]  # mid mcp
        pred_align = prj.copy()
        for i in range(prj.shape[0]):

            pred_ref_bone_len = np.linalg.norm(prj[i][ref_bone_link[0]] - prj[i][ref_bone_link[1]])
            gt_ref_bone_len = np.linalg.norm(gtj[i][ref_bone_link[0]] - gtj[i][ref_bone_link[1]])
            scale = gt_ref_bone_len / pred_ref_bone_len

            for j in range(21):
                pred_align[i][j] = gtj[i][root_idx] + scale * (prj[i][j] - prj[i][root_idx])

        return gtj, pred_align

    if key in ["do", "eo"]:
        # gtj :B*5*3
        # prj :B*5*3

        prj_ = prj.copy()[:, [4, 8, 12, 16, 20], :]  # B*5*3

        gtj_valid = []
        prj_valid_align = []

        for i in range(prj_.shape[0]):
            # 5*3
            mask = ~(np.isnan(gtj[i][:, 0]))
            if mask.sum() < 2:
                continue

            prj_mask = prj_[i][mask]  # m*3
            gtj_mask = gtj[i][mask]  # m*3

            gtj_valid_center = np.mean(gtj_mask, 0)
            prj_valid_center = np.mean(prj_mask, 0)

            gtj_center_length = np.linalg.norm(gtj_mask - gtj_valid_center, axis=1).mean()
            prj_center_length = np.linalg.norm(prj_mask - prj_valid_center, axis=1).mean()
            scale = gtj_center_length / prj_center_length

            prj_valid_align_i = gtj_valid_center + scale * (prj_[i][mask] - prj_valid_center)

            gtj_valid.append(gtj_mask)
            prj_valid_align.append(prj_valid_align_i)

        return np.array(gtj_valid), np.array(prj_valid_align)