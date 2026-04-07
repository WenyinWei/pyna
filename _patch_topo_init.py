from pathlib import Path
p = Path(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\__init__.py')
text = p.read_text(encoding='utf-8', errors='replace')
text = text.replace('from pyna.topo.island_chain import (\n    IslandChainOrbit,\n    ChainFixedPoint,\n)\n',
                    'from pyna.topo.island_chain import (\n    IslandChainOrbit,\n    ChainFixedPoint,\n)\nfrom pyna.plot.island import plot_island, island_section_points\nfrom pyna.plot.island_chain import plot_island_chain, island_chain_section_points\n')
text = text.replace('    "ChainFixedPoint",\n    # island-constrained healed coordinates\n',
                    '    "ChainFixedPoint",\n    # generic island / island-chain plotting\n    "plot_island",\n    "island_section_points",\n    "plot_island_chain",\n    "island_chain_section_points",\n    # island-constrained healed coordinates\n')
p.write_text(text, encoding='utf-8')
print('patched')
