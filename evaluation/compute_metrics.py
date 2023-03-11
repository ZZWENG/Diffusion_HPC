import sys

from cleanfid import fid


def evaluate_performance(real_img_path, gen_img_path):
    fid_score = fid.compute_fid(real_img_path, gen_img_path)
    kid_score = fid.compute_kid(real_img_path, gen_img_path)
    results = {
        'fid': fid_score, 
        'kid': kid_score
    }
    return results


if __name__ == '__main__':
    real_image_path = sys.argv[1]
    gen_image_path = sys.argv[2]
    results = evaluate_performance(real_img_path=real_image_path, gen_img_path=gen_image_path)
    print(results)
