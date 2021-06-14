'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de


About this file:
================
This file defines a wrapper for the loading functions of the SMPL+H model. 

Modules included:
- load_model:
  loads the SMPL+H model from a given file location (i.e. a .pkl file location), 
  or a dictionary object.

'''

def ready_arguments(fname_or_dict, posekey4vposed='pose'):
    import numpy as np
    import cPickle as pickle
    import chumpy as ch
    from chumpy.ch import MatVecMult
    from posemapper import posemap

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1]*3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    assert(posekey4vposed in dd)
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas'])+dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd[posekey4vposed]))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd[posekey4vposed]))

    return dd


def load_model(fname_or_dict='./models/SMPLH_female.pkl', ncomps=12, flat_hand_mean=False, v_template=None):
    ''' This model loads the fully articulable SMPL model,
    and replaces the 156-prefix last DOFS by ncomps from PCA'''

    from verts import verts_core
    import numpy as np
    import chumpy as ch
    import pickle
    import scipy.sparse as sp
    np.random.seed(1)

    if not isinstance(fname_or_dict, dict):
        smpl_data = pickle.load(open(fname_or_dict))
    else:
        smpl_data = fname_or_dict

    body_pose_dofs = 66

    from pickle import load
    with open('/is/ps2/dtzionas/mano/models/MANO_LEFT.pkl', 'rb') as f:
        hand_l = load(f)
    with open('/is/ps2/dtzionas/mano/models/MANO_RIGHT.pkl', 'rb') as f:
        hand_r = load(f)
    hands_componentsl = hand_l['hands_components']
    hands_meanl       = np.zeros(hands_componentsl.shape[1]) if flat_hand_mean else hand_l['hands_mean']
    hands_coeffsl     = hand_l['hands_coeffs'][:, :ncomps//2]
    hands_componentsr = hand_r['hands_components']
    hands_meanr       = np.zeros(hands_componentsr.shape[1]) if flat_hand_mean else hand_r['hands_mean']
    hands_coeffsr     = hand_r['hands_coeffs'][:, :ncomps//2]

    selected_components = np.vstack((np.hstack((hands_componentsl[:ncomps//2], np.zeros_like(hands_componentsl[:ncomps//2]))),
                                     np.hstack((np.zeros_like(hands_componentsr[:ncomps//2]), hands_componentsr[:ncomps//2]))))
    hands_mean = np.concatenate((hands_meanl, hands_meanr))

    pose_coeffs = ch.zeros(body_pose_dofs + selected_components.shape[0])
    full_hand_pose = pose_coeffs[body_pose_dofs:(body_pose_dofs+ncomps)].dot(selected_components)

    smpl_data['fullpose'] = ch.concatenate((pose_coeffs[:body_pose_dofs], hands_mean + full_hand_pose))
    smpl_data['pose'] = pose_coeffs

    Jreg = smpl_data['J_regressor']
    if not sp.issparse(Jreg):
        smpl_data['J_regressor'] = (sp.csc_matrix((Jreg.data, (Jreg.row, Jreg.col)), shape=Jreg.shape))
    # very slightly modify ready_arguments to make sure that it uses the fullpose (which will NOT be pose) for the computation of posedirs
    dd = ready_arguments(smpl_data, posekey4vposed='fullpose')

    # create the smpl formula with the fullpose,
    # but expose the PCA coefficients as smpl.pose for compatibility
    args = {
        'pose': dd['fullpose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style'],
    }

    result_previous, meta = verts_core(**args)
    result = result_previous + dd['trans'].reshape((1, 3))
    result.no_translation = result_previous

    if meta is not None:
        for field in ['Jtr', 'A', 'A_global', 'A_weighted']:
            if(hasattr(meta, field)):
                setattr(result, field, getattr(meta, field))

    if hasattr(result, 'Jtr'):
        result.J_transformed = result.Jtr + dd['trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    if v_template is not None:
        result.v_template[:] = v_template

    return result

if __name__ == '__main__':
    m = load_model()
    m.J_transformed
    print 'FINITO'
