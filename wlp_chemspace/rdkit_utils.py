import numpy as np
from rdkit.Chem import Descriptors, MolFromSmiles, MolFromSmarts
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Crippen
from rdkit.Chem import AllChem as Chem
def basic_descriptors_from_smiles(smiles: str) -> dict[str, float | None]:
    """
    Calculate basic molecular descriptors for a given SMILES string.

    Parameters:
    smiles (str): A SMILES string representing the molecule.

    Returns:
    dict: A dictionary containing the calculated descriptors.
    """
    mol=MolFromSmiles(smiles)
    dic={
        f"NumAromaticRings": Descriptors.NumAromaticRings(mol) if mol is not None else None,
        f"NumHAcceptors":Descriptors.NumHAcceptors(mol) if mol is not None else None,
        f"NumHDonors":Descriptors.NumHDonors(mol) if mol is not None else None,
        f"NumRotatableBonds":Descriptors.NumRotatableBonds(mol) if mol is not None else None,
        f"MW":Descriptors.MolWt(mol) if mol is not None else None,
        f"TPSA":Descriptors.TPSA(mol) if mol is not None else None,
        f"CrippenLogP":Crippen.MolLogP(mol) if mol is not None else None,
    }
    return dic

# define function that transforms SMILES strings into ECFPs
def ECFP_from_smiles(smiles,R = 2,L = 2048,use_features = True,use_chirality = True):
    """
    Extended Connectivity Fingerprints from SMILES strings using RDKit.
    Inputs:
    
    - smiles ... SMILES string of input compound
    - R ... maximum radius of circular substructures
    - L ... fingerprint-length
    - use_features ... if false then use standard DAYLIGHT atom features, if true then use pharmacophoric atom features
    - use_chirality ... if true then append tetrahedral chirality flags to atom features
    
    Outputs:
    - np.array(feature_list) ... ECFP with length L and maximum radius R
    """
    
    molecule = MolFromSmiles(smiles)
    feature_list = Chem.GetMorganFingerprintAsBitVect(molecule, radius=R, nBits=L, useFeatures=use_features, useChirality=use_chirality)
    return np.array(feature_list)

PATT=Chem.MolFromSmarts("[$([D1]=[*])]")
REPL=Chem.MolFromSmarts("[*]")
SCAFFOLD_TYPES=["rdkit","true_bm","bajorath"]
def scaffold_from_smiles(smiles,scaffold_type: str):
    """
    Generate a scaffold from a SMILES string based on the specified scaffold type.
    Adapted from https://github.com/rdkit/rdkit/discussions/6844
    Inputs:
    - smiles ... SMILES string of input compound
    - scaffold_type ... type of scaffold to generate; must be one of "rdkit", "true_bm", or "bajorath"
    Outputs:
    - scaff ... RDKit Mol object representing the generated scaffold
    Raises:
    - ValueError: If the scaffold_type is not one of the allowed types.
    """
    if scaffold_type not in SCAFFOLD_TYPES:
        raise ValueError(f"scaffold_type must be one of {SCAFFOLD_TYPES}")
    mol = MolFromSmiles(smiles)
    scaff=MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold_type == "bajorath":
        scaff=Chem.DeleteSubstructs(scaff, PATT)
    if scaffold_type == "true_bm":
        scaff=Chem.ReplaceSubstructs(scaff,PATT,REPL,replaceAll=True)[0]                                          
    return scaff