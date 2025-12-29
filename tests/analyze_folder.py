from traitly.internal_structure import FruitAnalyzer

path_folder = 'tests/sample_data/'

analyzer = FruitAnalyzer(path_folder)
analyzer.analyze_folder()
