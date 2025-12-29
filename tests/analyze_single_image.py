from traitly.internal_structure import FruitAnalyzer



path_image = "tests/sample_data/DP14-106.jpg"
analyzer = FruitAnalyzer(path_image)
analyzer.read_image() 
analyzer.setup_measurements()
analyzer.create_mask()
analyzer.find_fruits()
analyzer.analyze_image()
analyzer.results.save_all(output_dir = "tests/output_single_image/")

