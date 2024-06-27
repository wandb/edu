# Molecule

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/molecule.py#L25-L241' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Wandb class for 3D Molecular data.

```python
Molecule(
    data_or_path: Union[str, 'TextIO'],
    caption: Optional[str] = None,
    **kwargs
) -> None
```

| Arguments |  |
| :--- | :--- |
|  `data_or_path` |  (string, io) Molecule can be initialized from a file name or an io object. |
|  `caption` |  (string) Caption associated with the molecule for display. |

## Methods

### `from_rdkit`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/molecule.py#L99-L163)

```python
@classmethod
from_rdkit(
    data_or_path: "RDKitDataType",
    caption: Optional[str] = None,
    convert_to_3d_and_optimize: bool = (True),
    mmff_optimize_molecule_max_iterations: int = 200
) -> "Molecule"
```

Convert RDKit-supported file/object types to wandb.Molecule.

| Arguments |  |
| :--- | :--- |
|  `data_or_path` |  (string, rdkit.Chem.rdchem.Mol) Molecule can be initialized from a file name or an rdkit.Chem.rdchem.Mol object. |
|  `caption` |  (string) Caption associated with the molecule for display. |
|  `convert_to_3d_and_optimize` |  (bool) Convert to rdkit.Chem.rdchem.Mol with 3D coordinates. This is an expensive operation that may take a long time for complicated molecules. |
|  `mmff_optimize_molecule_max_iterations` |  (int) Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule |

### `from_smiles`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/molecule.py#L165-L202)

```python
@classmethod
from_smiles(
    data: str,
    caption: Optional[str] = None,
    sanitize: bool = (True),
    convert_to_3d_and_optimize: bool = (True),
    mmff_optimize_molecule_max_iterations: int = 200
) -> "Molecule"
```

Convert SMILES string to wandb.Molecule.

| Arguments |  |
| :--- | :--- |
|  `data` |  (string) SMILES string. |
|  `caption` |  (string) Caption associated with the molecule for display |
|  `sanitize` |  (bool) Check if the molecule is chemically reasonable by the RDKit's definition. |
|  `convert_to_3d_and_optimize` |  (bool) Convert to rdkit.Chem.rdchem.Mol with 3D coordinates. This is an expensive operation that may take a long time for complicated molecules. |
|  `mmff_optimize_molecule_max_iterations` |  (int) Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule |

| Class Variables |  |
| :--- | :--- |
|  `SUPPORTED_RDKIT_TYPES`<a id="SUPPORTED_RDKIT_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |
