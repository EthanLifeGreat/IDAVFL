import utils.CV as CV
from algorithm.encoder_matching_compare import *
from algorithm.encoder_matching_compare import encoder_matching_compare
from args import *
import time

# arguments = BCW_Arguments()
# arguments = DCC_Arguments()
# arguments = EPS5_Arguments()
arguments = HAR_Arguments()

# run_type = 'simple'
# run_type = 'simple_baseline'
# run_type = 'simple_compare'
# run_type = 'CV'
# run_type = 'CV_baseline'
run_type = 'CV_compare'
# run_type = 'CV_compare_10'
# run_type = 'CV_argument'

if __name__ == '__main__':

    time_start = time.time()

    if run_type == 'simple':
        print("running encoder_matching\n...\n")
        encoder_matching(arguments)

    elif run_type == 'simple_baseline':
        print("running encoder_matching_baseline\n...\n")
        encoder_matching_baseline(arguments)

    elif run_type == 'simple_compare':
        print("running encoder_matching_compare\n...\n")
        encoder_matching_compare(arguments)

    elif run_type == 'CV':
        print("running algorithm CV mode\n...\n")
        rec_name = "CV"
        CV.record_cv(rec_name, arguments, encoder_matching_train, encoder_matching_test)

    elif run_type == 'CV_baseline':
        print("running encoder_matching_baseline CV mode\n...\n")
        rec_name = "CV_baseline"
        CV.record_cv(rec_name, arguments, encoder_matching_train_baseline, encoder_matching_test_baseline)

    elif run_type == 'CV_compare':
        print("running encoder_matching_compare CV mode\n...\n")
        rec_name = "CV_compare"
        CV.record_cv(rec_name, arguments, encoder_matching_train_compare, encoder_matching_test_compare)
        if arguments.record_classification:
            res_name = "classification_result.csv"
            print("Recording classification result in " + res_name + "\n")
            arguments.data_frame.to_csv(arguments.rec_path + res_name, index_label="ID")
            print("Done!\n")

    elif run_type == 'CV_compare_10':
        print("running 10 times encoder_matching_compare CV mode\n...\n")
        rec_name = "10_times_CV_compare"
        CV.record_cv(rec_name, arguments, encoder_matching_train_compare, encoder_matching_test_compare, 10)
        if arguments.record_classification:
            res_name = "classification_result.csv"
            print("Recording classification result in " + res_name + "\n")
            arguments.data_frame.to_csv(arguments.rec_path + res_name, index_label="ID")
            print("Done!\n")

    elif run_type == 'CV_argument':
        print("running encoder_matching argument CV test mode\n...\n")
        arg_name = 'lam'
        arg_list = [.9, .7, .5, .3, .1]
        times = 5
        CV.record_cv_arg(arg_name, arguments, encoder_matching_train, encoder_matching_test, arg_list, times=times)
    else:
        print("Error: run_type error")

    time_end = time.time()
    print('Time Elapsed:{:.2f} min'.format((time_end - time_start) / 60))

