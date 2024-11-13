import json

def get_form_factor_and_total_density(system):
    """
    
    Returns form factor and total density profiles of the simulation
    
    FFsim: Form factor. Length 1000
    TDsim: Total density in pairs of (x,y). Maximum length 200
    """
    FFpathSIM = "./Databank/Data/Simulations/" + system['path'] + "FormFactor.json"
    TDpathSIM = "./Databank/Data/Simulations/" + system['path'] + "TotalDensity.json"

    # Load form factor and total density
    try:
        with open(FFpathSIM, "r") as json_file:
            FFsim = json.load(json_file)
        with open(TDpathSIM, "r") as json_file:
            TDsim = json.load(json_file)
    except Exception: 
        FFsim = None
        TDsim = None
    
    return FFsim, TDsim

