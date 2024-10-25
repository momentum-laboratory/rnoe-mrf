"""
an rNOE-MRF method aimed at the rapid acquisition and simultaneous creation of quantitative rNOE and semisolid MT maps
using two fully connected neural networks. 

Notes: 1 x BSA phantom, 4 x glycogen phantoms, and 7 x WT mouse data are provided. Human NN are provided, but no data due to subject confidentiality.
"""

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="The main file for the project")
    parser.add_argument("--name_of_scenario", help="Can be Liver_Glycogen_Phantoms-> 0 /BSA_Phantoms -> 1/ Mice ->2 human -> 3", type=int, default=1)
    parser.add_argument("--paper_example", help="Is it new data: 1 or the paper example: 0", type=int, default=0)

    # ----------------- Phantom Glycogen -----------------#
    parser.add_argument("--path_to_acquired_data" , help="The path to the acquired data in .mat format, when the data is in 'data' key", type=str, default="acquired_data_Bovine_150_200_300_15.mat")
    parser.add_argument("--name_of_quant_maps", help="The name of the quant maps", type=str, default="quant_maps_Bovine_150_200_300_15.mat")

    # ----------------- BSA -----------------#
    parser.add_argument("--path_to_acquired_data_bsa", help="The path to the acquired data in .mat format, when the data is in 'acquired_data' key", type=str, default="acquired_data_bsa.mat")
    parser.add_argument("--name_of_quant_maps_bsa", help="The name of the quant maps", type=str, default="quant_maps_bsa.mat")

    # ----------------- Mice-----------------#
    parser.add_argument("--path_to_acquired_data_mt", help="The path to the acquired data for MT in .mat format, when the data is in 'DATA' key", type=str, default="acquired_data_1_mt.mat")
    parser.add_argument("--path_to_acquired_data_noe", help="The path to the acquired data for NOE in .mat format, when the data is in 'DATA' key", type=str, default="acquired_data_1_noe.mat")
    parser.add_argument("--name_of_quant_maps_mt", help="The name of the quant maps", type=str, default="quant_maps_1_mt.mat")
    parser.add_argument("--name_of_quant_maps_noe", help="The name of the quant maps", type=str, default="quant_maps_1_noe.mat")

    # ----------------- Human -----------------#
    parser.add_argument("--path_to_acquired_data_human_mt", help="The path to the acquired data in .mat format, when the data is in 'data' key", type=str, default="MT_val8.mat")
    parser.add_argument("--path_to_acquired_data_human_noe", help="The path to the acquired data in .mat format, when the data is in 'data' key", type=str, default="rnoe_val8.mat")
    parser.add_argument("--dict_info", help="The path to the dictionary file", type=str, default="noe_mat")
    parser.add_argument("--name_of_quant_maps_human_mt", help="The name of the quant maps", type=str, default="quant_maps_mt.mat")
    parser.add_argument("--name_of_quant_maps_human_noe", help="The name of the quant maps", type=str, default="quant_maps_noe.mat")

    args = parser.parse_args()

    ###########################################################
    # Main
    ###########################################################

    if args.name_of_scenario == 0:
        from src.Liver_Glycogen_Phantoms.main import main
    elif args.name_of_scenario == 1:
        from src.BSA_Phantoms.main import main
    elif args.name_of_scenario == 2:
        from src.Mice.main import main
    elif args.name_of_scenario == 3:
        from src.Human.main import main
        
    else:
        raise ValueError("Undefined scenario")

    main(args)
