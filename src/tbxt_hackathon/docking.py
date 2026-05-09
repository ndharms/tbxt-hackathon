"""AutoDock Vina docking pipeline for TBXT pocket scoring.

Two modes:
  - dock_single(): detailed report with interaction analysis and 3D visualization
  - dock_batch(): fast scoring of many SMILES for compound filtering

Receptor preparation uses a manual PDB-to-PDBQT converter (no OpenBabel or
ADFR Suite required). Ligand preparation uses Meeko + RDKit conformer generation.

Example:
    >>> from tbxt_hackathon.docking import dock_single, dock_batch
    >>> report = dock_single("CC(=O)Nc1ccccc1", "data/structures/TGT_TBXT_A_pocket.pdb")
    >>> report.show_3d()
    >>> print(report.summary())
    >>> df = dock_batch(["CC(=O)Nc1ccccc1", "c1ccccc1"], "data/structures/TGT_TBXT_A_pocket.pdb")
"""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Interaction:
    type: str
    protein_residue: str
    protein_atom: str
    ligand_atom_idx: int
    distance: float
    angle: float | None = None
    detail: str = ""


@dataclass
class PoseResult:
    """A single docked pose from one conformer run."""
    conformer_idx: int
    pose_idx: int
    score: float
    inter_energy: float
    intra_energy: float
    pose_pdb: str
    pose_sdf: str
    interactions: list[Interaction]

    def summary(self) -> str:
        lines = [
            f"  Conformer {self.conformer_idx}, pose {self.pose_idx}:",
            f"    Score:         {self.score:.2f} kcal/mol",
            f"    Inter:         {self.inter_energy:.2f} kcal/mol",
            f"    Intra:         {self.intra_energy:.2f} kcal/mol",
            f"    Interactions:  {len(self.interactions)}",
        ]
        for ix in self.interactions:
            angle_str = f", angle={ix.angle:.0f}" if ix.angle is not None else ""
            lines.append(
                f"      {ix.type:<14s} {ix.protein_residue:<8s} "
                f"{ix.protein_atom:<4s}  {ix.distance:.2f} A{angle_str}"
                + (f"  ({ix.detail})" if ix.detail else "")
            )
        return "\n".join(lines)


@dataclass
class DockingReport:
    smiles: str
    score: float
    ligand_efficiency: float
    n_heavy_atoms: int
    mw: float
    poses: list[PoseResult]
    receptor_pdb: str
    box_center: tuple[float, float, float]
    box_size: tuple[float, float, float]
    n_conformers: int = 3

    @property
    def best_pose(self) -> PoseResult:
        return min(self.poses, key=lambda p: p.score)

    @property
    def interactions(self) -> list[Interaction]:
        return self.best_pose.interactions

    def summary(self) -> str:
        bp = self.best_pose
        scores = [p.score for p in self.poses]
        lines = [
            f"Docking report: {self.smiles}",
            f"  Best score:        {self.score:.2f} kcal/mol  "
            f"(conformer {bp.conformer_idx}, pose {bp.pose_idx})",
            f"  Inter-molecular:   {bp.inter_energy:.2f} kcal/mol",
            f"  Intra-molecular:   {bp.intra_energy:.2f} kcal/mol",
            f"  Ligand efficiency: {self.ligand_efficiency:.3f} kcal/mol/HA",
            f"  Heavy atoms:       {self.n_heavy_atoms}",
            f"  MW:                {self.mw:.1f}",
            f"  Conformers run:    {self.n_conformers}",
            f"  Total poses:       {len(self.poses)}",
            f"  Score range:       {min(scores):.2f} to {max(scores):.2f} kcal/mol",
            "",
            f"  Best-pose interactions ({len(bp.interactions)}):",
        ]
        for ix in bp.interactions:
            angle_str = f", angle={ix.angle:.0f}" if ix.angle is not None else ""
            lines.append(
                f"    {ix.type:<14s} {ix.protein_residue:<8s} "
                f"{ix.protein_atom:<4s}  {ix.distance:.2f} A{angle_str}"
                + (f"  ({ix.detail})" if ix.detail else "")
            )
        return "\n".join(lines)

    def show_3d(self, pose_idx: int | None = None, width: int = 900, height: int = 600):
        """Interactive py3Dmol visualization in Jupyter.

        Args:
            pose_idx: Index into self.poses to display. None = best pose.
        """
        pose = self.poses[pose_idx] if pose_idx is not None else self.best_pose
        return _render_pose_3d(
            self.receptor_pdb, pose.pose_pdb, pose.pose_sdf, pose.interactions,
            self.box_center, width, height,
        )

    def show_all_poses(self, width: int = 900, height: int = 600):
        """Overlay all poses in a single view, best pose highlighted."""
        import py3Dmol

        view = py3Dmol.view(width=width, height=height)
        view.addModel(self.receptor_pdb, "pdb")
        view.setStyle(
            {"model": 0},
            {"cartoon": {"color": "spectrum", "opacity": 0.85}},
        )

        best = self.best_pose
        for i, pose in enumerate(self.poses):
            view.addModel(pose.pose_sdf, "sdf")
            model_idx = i + 1
            if pose is best:
                view.setStyle(
                    {"model": model_idx},
                    {
                        "stick": {"colorscheme": "greenCarbon", "radius": 0.15},
                        "sphere": {"colorscheme": "greenCarbon", "scale": 0.3},
                    },
                )
            else:
                view.setStyle(
                    {"model": model_idx},
                    {"stick": {"color": "0xAAAAAA", "radius": 0.1, "opacity": 0.4}},
                )

        cx, cy, cz = self.box_center
        view.zoomTo({"center": {"x": cx, "y": cy, "z": cz}})
        return view.show()


def _render_pose_3d(
    receptor_pdb: str,
    pose_pdb: str,
    pose_sdf: str,
    interactions: list[Interaction],
    box_center: tuple[float, float, float],
    width: int = 900,
    height: int = 600,
):
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)

    view.addModel(receptor_pdb, "pdb")
    view.setStyle(
        {"model": 0},
        {"cartoon": {"color": "spectrum", "opacity": 0.7}},
    )

    view.addModel(pose_sdf, "sdf")
    view.setStyle(
        {"model": 1},
        {
            "stick": {"colorscheme": "greenCarbon", "radius": 0.15},
            "sphere": {"colorscheme": "greenCarbon", "scale": 0.3},
        },
    )

    interacting_residues = {
        (ix.protein_residue[:3], ix.protein_residue[3:])
        for ix in interactions
    }
    for res_name, res_seq in interacting_residues:
        view.addStyle(
            {"model": 0, "resn": res_name, "resi": res_seq},
            {"stick": {"colorscheme": "whiteCarbon", "radius": 0.12}},
        )
        view.addResLabels(
            {"model": 0, "resn": res_name, "resi": res_seq, "atom": "CA"},
            {
                "fontSize": 11,
                "fontColor": "white",
                "backgroundColor": "0x333333",
                "backgroundOpacity": 0.7,
                "showBackground": True,
            },
        )

    color_map = {
        "hbond": "0xFFD700",
        "pi_stacking": "0x00CC66",
        "hydrophobic": "0x888888",
        "salt_bridge": "0xFF4444",
    }

    for ix in interactions:
        p_xyz = _residue_atom_xyz(receptor_pdb, ix.protein_residue, ix.protein_atom)
        l_xyz = _ligand_atom_xyz(pose_pdb, ix.ligand_atom_idx)
        if p_xyz is None or l_xyz is None:
            continue

        color = color_map.get(ix.type, "0xAAAAAA")

        view.addCylinder(
            {
                "start": {"x": p_xyz[0], "y": p_xyz[1], "z": p_xyz[2]},
                "end": {"x": l_xyz[0], "y": l_xyz[1], "z": l_xyz[2]},
                "color": color,
                "radius": 0.06,
                "dashed": True,
                "dashLength": 0.2,
                "gapLength": 0.15,
            }
        )

        view.addLabel(
            f"{ix.type} {ix.distance:.1f}A",
            {
                "position": {
                    "x": (p_xyz[0] + l_xyz[0]) / 2,
                    "y": (p_xyz[1] + l_xyz[1]) / 2,
                    "z": (p_xyz[2] + l_xyz[2]) / 2,
                },
                "fontSize": 10,
                "backgroundColor": "black",
                "fontColor": "white",
                "backgroundOpacity": 0.6,
            },
        )

    cx, cy, cz = box_center
    view.zoomTo({"center": {"x": cx, "y": cy, "z": cz}})
    return view.show()


# ---------------------------------------------------------------------------
# Receptor preparation (PDB → PDBQT, no external tools)
# ---------------------------------------------------------------------------

_AD4_PROTEIN_TYPES = {
    "C": "C",
    "N": "NA",
    "O": "OA",
    "S": "SA",
    "H": "HD",
    "F": "F",
    "P": "P",
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "FE": "Fe",
    "ZN": "Zn",
    "MG": "Mg",
    "MN": "Mn",
    "CA": "Ca",
    "SE": "Se",
}

_NON_ACCEPTOR_N = {
    ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"),
    ("LYS", "NZ"),
}


def _element_from_pdb_line(line: str) -> str:
    elem = line[76:78].strip().upper()
    if elem:
        return elem
    name = line[12:16].strip()
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return "C"


def _ad4_type_for_protein_atom(element: str, atom_name: str, res_name: str) -> str:
    el = element.upper()
    if el == "N":
        if atom_name == "N":
            return "N"
        if (res_name, atom_name) in _NON_ACCEPTOR_N:
            return "N"
        return "NA"
    if el == "O":
        return "OA"
    if el == "S":
        return "SA"
    if el == "H":
        return "HD"
    return _AD4_PROTEIN_TYPES.get(el, el)


def _pdb_line_to_pdbqt(line: str) -> str:
    element = _element_from_pdb_line(line)
    atom_name = line[12:16].strip()
    res_name = line[17:20].strip()
    ad_type = _ad4_type_for_protein_atom(element, atom_name, res_name)
    base = line[:54]
    occ_bfac = line[54:66] if len(line) >= 66 else "  1.00  0.00"
    return f"{base}{occ_bfac}    +0.000 {ad_type:<2s}\n"


def prepare_receptor(pocket_pdb: str | Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Parse pocket PDB, strip ligand, return (PDBQT string, box_center, box_size).

    The search box is derived from the existing ligand coordinates with 10 Å padding.
    """
    pocket_pdb = Path(pocket_pdb)
    lines = pocket_pdb.read_text().splitlines(keepends=True)

    protein_lines: list[str] = []
    ligand_coords: list[list[float]] = []

    for line in lines:
        if line.startswith("ATOM"):
            protein_lines.append(line)
        elif line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ligand_coords.append([x, y, z])
            except ValueError:
                pass

    pdbqt_lines = [_pdb_line_to_pdbqt(ln) for ln in protein_lines]
    pdbqt_str = "".join(pdbqt_lines)

    lig_arr = np.array(ligand_coords)
    center = lig_arr.mean(axis=0)
    extent = lig_arr.max(axis=0) - lig_arr.min(axis=0)
    padding = 10.0
    box_size = extent + padding

    return pdbqt_str, center, box_size


def _receptor_pdb_block(pocket_pdb: str | Path) -> str:
    """Return just the protein ATOM records as a PDB block (for visualization)."""
    lines = Path(pocket_pdb).read_text().splitlines(keepends=True)
    return "".join(ln for ln in lines if ln.startswith(("ATOM", "TER", "END")))


# ---------------------------------------------------------------------------
# Ligand preparation (SMILES → PDBQT via RDKit + Meeko)
# ---------------------------------------------------------------------------

def prepare_ligand(
    smiles: str, random_seed: int = 42,
) -> tuple[str, Chem.Mol, list[int]]:
    """Generate 3D conformer and convert to PDBQT string via Meeko.

    Returns (pdbqt_string, rdkit_mol_with_3d_coords, pdbqt_to_rdkit_map).
    The map gives the RDKit atom index for each heavy atom in PDBQT order.
    Raises ValueError if molecule cannot be prepared.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            raise ValueError(f"Cannot generate 3D conformer for: {smiles}")

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(
        mol_setups[0], add_index_map=True,
    )
    if not is_ok:
        raise ValueError(f"Meeko PDBQT preparation failed for {smiles}: {err_msg}")

    pdbqt_to_rdkit = _parse_index_map(pdbqt_string, mol)

    return pdbqt_string, mol, pdbqt_to_rdkit


def _parse_index_map(pdbqt_string: str, mol: Chem.Mol) -> list[int]:
    """Extract the PDBQT->RDKit heavy-atom index map from Meeko's REMARK INDEX MAP.

    Returns a list where position i is the RDKit atom index for PDBQT heavy atom i.
    """
    all_tokens: list[str] = []
    for line in pdbqt_string.splitlines():
        if line.startswith("REMARK INDEX MAP"):
            all_tokens.extend(line.split()[3:])

    if not all_tokens:
        return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

    pairs = list(zip(all_tokens[0::2], all_tokens[1::2]))
    serial_to_rdkit = {
        int(pdbqt_serial): int(rdkit_1based) - 1
        for rdkit_1based, pdbqt_serial in pairs
    }

    heavy_serials = []
    for atom_line in pdbqt_string.splitlines():
        if not atom_line.startswith(("ATOM", "HETATM")):
            continue
        ad_type = atom_line[77:79].strip() if len(atom_line) >= 79 else ""
        if ad_type in ("H", "HD", "HS"):
            continue
        serial = int(atom_line[6:11])
        heavy_serials.append(serial)

    return [serial_to_rdkit[s] for s in heavy_serials]


# ---------------------------------------------------------------------------
# Docking engine
# ---------------------------------------------------------------------------

def _run_vina(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: np.ndarray,
    box_size: np.ndarray,
    exhaustiveness: int = 32,
    n_poses: int = 9,
) -> tuple[np.ndarray, str]:
    """Run Vina docking, return (energies_array, poses_pdbqt_string)."""
    from vina import Vina

    v = Vina(sf_name="vina", cpu=0, verbosity=0)

    with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as f:
        f.write(receptor_pdbqt)
        receptor_path = f.name

    try:
        v.set_receptor(rigid_pdbqt_filename=receptor_path)
    finally:
        Path(receptor_path).unlink(missing_ok=True)

    v.compute_vina_maps(
        center=center.tolist(),
        box_size=box_size.tolist(),
    )

    v.set_ligand_from_string(ligand_pdbqt)

    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    energies = v.energies()
    poses_pdbqt = v.poses()
    return energies, poses_pdbqt


# ---------------------------------------------------------------------------
# Interaction detection
# ---------------------------------------------------------------------------

@dataclass
class _Atom3D:
    idx: int
    name: str
    element: str
    coords: np.ndarray
    residue: str = ""
    is_aromatic: bool = False
    is_donor: bool = False
    is_acceptor: bool = False


_ACCEPTOR_ATOMS = {
    "N", "O", "F", "S",
}

_PROTEIN_DONOR_ATOMS = {
    ("N",),
    ("NZ",), ("NH1",), ("NH2",), ("NE",), ("NE1",), ("NE2",), ("ND1",), ("ND2",),
    ("OG",), ("OG1",), ("OH",),
}

_AROMATIC_RESIDUES_RINGS = {
    "PHE": [("CG", "CD1", "CE1", "CZ", "CE2", "CD2")],
    "TYR": [("CG", "CD1", "CE1", "CZ", "CE2", "CD2")],
    "TRP": [
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CE2", "CD2", "CE3", "CZ3", "CH2", "CZ2"),
    ],
    "HIS": [("CG", "ND1", "CE1", "NE2", "CD2")],
}


def _parse_protein_atoms(pdb_block: str) -> list[_Atom3D]:
    atoms = []
    for line in pdb_block.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue

        name = line[12:16].strip()
        res_name = line[17:20].strip()
        res_seq = line[22:26].strip()
        element = line[76:78].strip().upper() if len(line) >= 78 else name[0]
        residue = f"{res_name}{res_seq}"

        is_donor = (name,) in _PROTEIN_DONOR_ATOMS
        is_acceptor = element in _ACCEPTOR_ATOMS

        atoms.append(_Atom3D(
            idx=len(atoms),
            name=name,
            element=element,
            coords=np.array([x, y, z]),
            residue=residue,
            is_donor=is_donor,
            is_acceptor=is_acceptor,
        ))
    return atoms


def _parse_ligand_atoms_from_mol(mol: Chem.Mol, conf_id: int = 0) -> list[_Atom3D]:
    conf = mol.GetConformer(conf_id)
    atoms = []
    ri = mol.GetRingInfo()
    aromatic_atoms = set()
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_atoms.update(ring)

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        el = atom.GetSymbol().upper()
        atoms.append(_Atom3D(
            idx=atom.GetIdx(),
            name=atom.GetSymbol() + str(atom.GetIdx()),
            element=el,
            coords=np.array([pos.x, pos.y, pos.z]),
            is_aromatic=atom.GetIdx() in aromatic_atoms,
            is_donor=el in ("N", "O", "S") and atom.GetTotalNumHs() > 0,
            is_acceptor=el in _ACCEPTOR_ATOMS,
        ))
    return atoms


def _ring_centroid_and_normal(coords: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array(coords)
    centroid = pts.mean(axis=0)
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm > 1e-6:
        normal = normal / norm
    return centroid, normal


def _get_protein_aromatic_rings(
    pdb_block: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return list of (residue_label, centroid, normal) for aromatic rings."""
    atom_lookup: dict[tuple[str, str], np.ndarray] = {}
    res_names: dict[str, str] = {}

    for line in pdb_block.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        name = line[12:16].strip()
        res_name = line[17:20].strip()
        res_seq = line[22:26].strip()
        key = f"{res_name}{res_seq}"
        atom_lookup[(key, name)] = np.array([x, y, z])
        res_names[key] = res_name

    rings = []
    for res_key, res_name in res_names.items():
        if res_name not in _AROMATIC_RESIDUES_RINGS:
            continue
        for ring_atoms in _AROMATIC_RESIDUES_RINGS[res_name]:
            coords = []
            for aname in ring_atoms:
                if (res_key, aname) in atom_lookup:
                    coords.append(atom_lookup[(res_key, aname)])
            if len(coords) >= 3:
                centroid, normal = _ring_centroid_and_normal(coords)
                rings.append((res_key, centroid, normal))
    return rings


def _get_ligand_aromatic_rings(
    mol: Chem.Mol, conf_id: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
    """Return list of (centroid, normal, atom_indices) for ligand aromatic rings."""
    conf = mol.GetConformer(conf_id)
    ri = mol.GetRingInfo()
    rings = []
    for ring in ri.AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        coords = []
        for idx in ring:
            pos = conf.GetAtomPosition(idx)
            coords.append(np.array([pos.x, pos.y, pos.z]))
        if len(coords) >= 3:
            centroid, normal = _ring_centroid_and_normal(coords)
            rings.append((centroid, normal, list(ring)))
    return rings


def detect_interactions(
    receptor_pdb: str,
    ligand_mol: Chem.Mol,
    conf_id: int = 0,
    hbond_dist: float = 3.5,
    pi_dist: float = 5.5,
    hydrophobic_dist: float = 4.0,
    salt_dist: float = 4.0,
) -> list[Interaction]:
    prot_atoms = _parse_protein_atoms(receptor_pdb)
    lig_atoms = _parse_ligand_atoms_from_mol(ligand_mol, conf_id)

    interactions: list[Interaction] = []

    prot_coords = np.array([a.coords for a in prot_atoms])
    lig_coords = np.array([a.coords for a in lig_atoms])

    if len(prot_coords) == 0 or len(lig_coords) == 0:
        return interactions

    dists = np.linalg.norm(prot_coords[:, None, :] - lig_coords[None, :, :], axis=2)

    # H-bonds: protein donor/acceptor ↔ ligand acceptor/donor
    for pi, pa in enumerate(prot_atoms):
        for li, la in enumerate(lig_atoms):
            d = dists[pi, li]
            if d > hbond_dist:
                continue
            if pa.is_donor and la.is_acceptor:
                interactions.append(Interaction(
                    type="hbond",
                    protein_residue=pa.residue,
                    protein_atom=pa.name,
                    ligand_atom_idx=la.idx,
                    distance=round(float(d), 2),
                    detail="protein_donor",
                ))
            elif pa.is_acceptor and la.is_donor:
                interactions.append(Interaction(
                    type="hbond",
                    protein_residue=pa.residue,
                    protein_atom=pa.name,
                    ligand_atom_idx=la.idx,
                    distance=round(float(d), 2),
                    detail="ligand_donor",
                ))

    # Pi-stacking
    prot_rings = _get_protein_aromatic_rings(receptor_pdb)
    lig_rings = _get_ligand_aromatic_rings(ligand_mol, conf_id)

    for pres, pc, pn in prot_rings:
        for lc, ln, latoms in lig_rings:
            d = float(np.linalg.norm(pc - lc))
            if d > pi_dist:
                continue
            cos_angle = abs(float(np.dot(pn, ln)))
            cos_angle = min(cos_angle, 1.0)
            angle = float(np.degrees(np.arccos(cos_angle)))
            if angle < 30:
                detail = "face_to_face"
            elif angle > 60:
                detail = "edge_to_face"
            else:
                continue
            interactions.append(Interaction(
                type="pi_stacking",
                protein_residue=pres,
                protein_atom="ring",
                ligand_atom_idx=latoms[0],
                distance=round(d, 2),
                angle=round(angle, 1),
                detail=detail,
            ))

    # Hydrophobic contacts (C–C)
    for pi, pa in enumerate(prot_atoms):
        if pa.element != "C":
            continue
        for li, la in enumerate(lig_atoms):
            if la.element != "C":
                continue
            d = dists[pi, li]
            if d <= hydrophobic_dist:
                interactions.append(Interaction(
                    type="hydrophobic",
                    protein_residue=pa.residue,
                    protein_atom=pa.name,
                    ligand_atom_idx=la.idx,
                    distance=round(float(d), 2),
                ))

    # Salt bridges (charged N ↔ charged O on Asp/Glu)
    acidic_oxygens = [
        a for a in prot_atoms
        if a.element == "O" and a.name.startswith("O") and a.residue[:3] in ("ASP", "GLU")
    ]
    basic_nitrogens = [
        a for a in lig_atoms
        if a.element == "N" and a.is_donor
    ]
    for pa in acidic_oxygens:
        for la in basic_nitrogens:
            d = float(np.linalg.norm(pa.coords - la.coords))
            if d <= salt_dist:
                interactions.append(Interaction(
                    type="salt_bridge",
                    protein_residue=pa.residue,
                    protein_atom=pa.name,
                    ligand_atom_idx=la.idx,
                    distance=round(d, 2),
                ))

    _deduplicate_hydrophobic(interactions)
    return interactions


def _deduplicate_hydrophobic(interactions: list[Interaction]) -> None:
    """Keep only the closest hydrophobic contact per residue."""
    best: dict[str, int] = {}
    to_remove = []
    for i, ix in enumerate(interactions):
        if ix.type != "hydrophobic":
            continue
        key = ix.protein_residue
        if key in best:
            prev = interactions[best[key]]
            if ix.distance < prev.distance:
                to_remove.append(best[key])
                best[key] = i
            else:
                to_remove.append(i)
        else:
            best[key] = i
    for idx in sorted(to_remove, reverse=True):
        interactions.pop(idx)


# ---------------------------------------------------------------------------
# Pose extraction helpers
# ---------------------------------------------------------------------------

def _nth_pose_pdbqt(poses_pdbqt: str, n: int = 0) -> str:
    """Extract the Nth pose (0-indexed) from a multi-model PDBQT string."""
    models: list[list[str]] = []
    current: list[str] = []
    for line in poses_pdbqt.splitlines(keepends=True):
        if line.startswith("MODEL"):
            current = []
        elif line.startswith("ENDMDL"):
            models.append(current)
        else:
            current.append(line)
    if not models and current:
        models.append(current)
    if n >= len(models):
        n = len(models) - 1
    return "".join(models[n])


_AD4_TYPE_TO_ELEMENT = {
    "C": "C", "A": "C",
    "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O",
    "S": "S", "SA": "S",
    "H": "H", "HD": "H", "HS": "H",
    "F": "F", "Cl": "Cl", "CL": "Cl",
    "Br": "Br", "BR": "Br", "I": "I",
    "P": "P", "Fe": "Fe", "Zn": "Zn",
    "Mg": "Mg", "Mn": "Mn", "Ca": "Ca",
    "Se": "Se",
}


def _pdbqt_to_pdb(pdbqt_str: str) -> str:
    """Convert ligand PDBQT to PDB for py3Dmol visualization.

    Uses HETATM records and maps AD4 atom types back to element symbols
    so py3Dmol renders the correct molecule.
    """
    pdb_lines = []
    serial = 1
    for line in pdbqt_str.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        ad_type = line[77:79].strip() if len(line) >= 79 else ""
        element = _AD4_TYPE_TO_ELEMENT.get(ad_type, "")
        if not element:
            name = line[12:16].strip()
            element = "".join(c for c in name if c.isalpha())[:2] or "C"

        coords = line[30:54]
        atom_name = line[12:16]
        pdb_lines.append(
            f"HETATM{serial:5d} {atom_name}LIG L   1    "
            f"{coords}  1.00  0.00          {element:>2s}  \n"
        )
        serial += 1
    pdb_lines.append("END\n")
    return "".join(pdb_lines)


def _update_ligand_mol_coords(
    mol: Chem.Mol,
    pdbqt_pose: str,
    pdbqt_to_rdkit: list[int],
    conf_id: int = 0,
) -> Chem.Mol:
    """Update RDKit mol coordinates from the docked PDBQT pose.

    Uses pdbqt_to_rdkit mapping (from Meeko's INDEX MAP) to assign each
    PDBQT heavy-atom coordinate to the correct RDKit atom index.
    """
    coords = []
    for line in pdbqt_pose.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        ad_type = line[77:79].strip() if len(line) >= 79 else ""
        if ad_type in ("H", "HD", "HS"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))
        except (ValueError, IndexError):
            continue

    if len(coords) != len(pdbqt_to_rdkit):
        warnings.warn(
            f"PDBQT heavy atom count ({len(coords)}) != index map length "
            f"({len(pdbqt_to_rdkit)}). Coordinate mapping may be incorrect.",
            stacklevel=2,
        )

    new_mol = Chem.RWMol(mol)
    conf = new_mol.GetConformer(conf_id)
    for i, rdkit_idx in enumerate(pdbqt_to_rdkit):
        if i >= len(coords):
            break
        conf.SetAtomPosition(rdkit_idx, Chem.rdGeometry.Point3D(*coords[i]))

    return new_mol.GetMol()


def _residue_atom_xyz(
    pdb_block: str, residue: str, atom_name: str,
) -> tuple[float, float, float] | None:
    res_name = residue[:3]
    res_seq = residue[3:]
    for line in pdb_block.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[17:20].strip() == res_name and line[22:26].strip() == res_seq:
            if line[12:16].strip() == atom_name:
                try:
                    return (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                except (ValueError, IndexError):
                    return None
    if atom_name == "ring":
        coords = []
        for line in pdb_block.splitlines():
            if not line.startswith("ATOM"):
                continue
            if line[17:20].strip() == res_name and line[22:26].strip() == res_seq:
                try:
                    coords.append((
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ))
                except (ValueError, IndexError):
                    pass
        if coords:
            arr = np.array(coords)
            c = arr.mean(axis=0)
            return (float(c[0]), float(c[1]), float(c[2]))
    return None


def _ligand_atom_xyz(
    pose_pdb: str, atom_idx: int,
) -> tuple[float, float, float] | None:
    heavy_count = 0
    for line in pose_pdb.splitlines():
        if not line.startswith("HETATM"):
            continue
        el = line[76:78].strip().upper() if len(line) >= 78 else ""
        if el == "H":
            continue
        if heavy_count == atom_idx:
            try:
                return (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except (ValueError, IndexError):
                return None
        heavy_count += 1
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dock_single(
    smiles: str,
    pocket_pdb: str | Path,
    n_conformers: int = 3,
    exhaustiveness: int = 32,
    n_poses: int = 9,
) -> DockingReport:
    """Dock a single compound with full interaction analysis and visualization.

    Runs multiple independent conformer attempts (each with a different random
    seed for 3D embedding). The best score across all conformers is the reported
    score. All poses from all conformers are kept for inspection.

    Args:
        smiles: Input SMILES string.
        pocket_pdb: Path to pocket PDB file (protein + reference ligand).
        n_conformers: Number of independent conformer/docking runs.
        exhaustiveness: Vina search exhaustiveness (higher = slower, more thorough).
        n_poses: Maximum number of poses per conformer run.

    Returns:
        DockingReport with all poses accessible via report.poses and
        report.show_3d(pose_idx=...) for stepping through them.
    """
    receptor_pdbqt, center, box_size = prepare_receptor(pocket_pdb)
    receptor_pdb = _receptor_pdb_block(pocket_pdb)

    mol_parsed = Chem.MolFromSmiles(smiles)
    if mol_parsed is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    n_ha = Descriptors.HeavyAtomCount(mol_parsed)
    mw = Descriptors.MolWt(mol_parsed)

    all_poses: list[PoseResult] = []

    for conf_i in range(n_conformers):
        seed = 42 + conf_i * 137
        ligand_pdbqt, lig_mol, pdbqt_to_rdkit = prepare_ligand(
            smiles, random_seed=seed,
        )

        energies, poses_pdbqt = _run_vina(
            receptor_pdbqt, ligand_pdbqt, center, box_size,
            exhaustiveness=exhaustiveness, n_poses=n_poses,
        )

        for pose_i in range(energies.shape[0]):
            pose_pdbqt = _nth_pose_pdbqt(poses_pdbqt, pose_i)
            pose_pdb = _pdbqt_to_pdb(pose_pdbqt)

            docked_mol = _update_ligand_mol_coords(
                lig_mol, pose_pdbqt, pdbqt_to_rdkit,
            )
            interactions = detect_interactions(receptor_pdb, docked_mol)
            pose_sdf = Chem.MolToMolBlock(Chem.RemoveAllHs(docked_mol))

            score = float(energies[pose_i, 0])
            inter = float(energies[pose_i, 1]) if energies.shape[1] > 1 else 0.0
            intra = float(energies[pose_i, 2]) if energies.shape[1] > 2 else 0.0

            all_poses.append(PoseResult(
                conformer_idx=conf_i,
                pose_idx=pose_i,
                score=score,
                inter_energy=inter,
                intra_energy=intra,
                pose_pdb=pose_pdb,
                pose_sdf=pose_sdf,
                interactions=interactions,
            ))

    best_score = min(p.score for p in all_poses)
    le = -best_score / n_ha if n_ha > 0 else 0.0

    return DockingReport(
        smiles=smiles,
        score=best_score,
        ligand_efficiency=le,
        n_heavy_atoms=n_ha,
        mw=mw,
        poses=all_poses,
        receptor_pdb=receptor_pdb,
        box_center=tuple(center.tolist()),
        box_size=tuple(box_size.tolist()),
        n_conformers=n_conformers,
    )


def dock_batch(
    smiles_list: list[str],
    pocket_pdb: str | Path,
    n_conformers: int = 3,
    exhaustiveness: int = 8,
    n_poses: int = 1,
) -> pd.DataFrame:
    """Batch-dock a list of SMILES and return scores for filtering.

    Runs n_conformers independent docking attempts per compound and reports
    the minimum (best) score. Uses lower exhaustiveness for speed.

    Args:
        smiles_list: List of SMILES strings to dock.
        pocket_pdb: Path to pocket PDB file (protein + reference ligand).
        n_conformers: Number of independent conformer/docking runs per compound.
        exhaustiveness: Vina search exhaustiveness (lower = faster).
        n_poses: Number of poses per conformer run (1 for scoring only).

    Returns:
        DataFrame with columns: smiles, score, ligand_efficiency, n_heavy_atoms,
        mw, score_std, status.
    """
    assert len(smiles_list) > 0, "smiles_list must not be empty"

    receptor_pdbqt, center, box_size = prepare_receptor(pocket_pdb)

    records = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"RDKit cannot parse SMILES: {smi}")
            n_ha = Descriptors.HeavyAtomCount(mol)
            mw = Descriptors.MolWt(mol)

            conf_scores = []
            for conf_i in range(n_conformers):
                seed = 42 + conf_i * 137
                ligand_pdbqt, _, _ = prepare_ligand(smi, random_seed=seed)
                energies, _ = _run_vina(
                    receptor_pdbqt, ligand_pdbqt, center, box_size,
                    exhaustiveness=exhaustiveness, n_poses=n_poses,
                )
                conf_scores.append(float(energies[0, 0]))

            score = min(conf_scores)
            le = -score / n_ha if n_ha > 0 else 0.0
            records.append({
                "smiles": smi,
                "score": round(score, 3),
                "ligand_efficiency": round(le, 4),
                "n_heavy_atoms": n_ha,
                "mw": round(mw, 2),
                "score_std": round(float(np.std(conf_scores)), 3),
                "status": "ok",
            })
        except Exception as exc:
            records.append({
                "smiles": smi,
                "score": float("nan"),
                "ligand_efficiency": float("nan"),
                "n_heavy_atoms": 0,
                "mw": 0.0,
                "score_std": float("nan"),
                "status": str(exc),
            })

    df = pd.DataFrame(records)
    assert df.shape[0] == len(smiles_list), (
        f"Output rows ({df.shape[0]}) != input SMILES ({len(smiles_list)})"
    )
    return df.sort_values("score", ascending=True, na_position="last").reset_index(drop=True)
