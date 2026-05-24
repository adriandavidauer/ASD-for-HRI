from .asd import Architecture_Options, DetectVVAD
# initialize all the models for DetectVVAD once so they will be automatically downloaded. 
for model in Architecture_Options:
    print(f"Initializing model {model} for DetectVVAD...")
    DetectVVAD(architecture=model)