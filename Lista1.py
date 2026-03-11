import numpy as np

SAMPLE_SIZES = [10, 50, 1000]

def generate_population(size: int, mean: float, std: float) -> np.ndarray:
	return np.random.normal(loc=mean, scale=std, size=size)

def random_select(population: np.ndarray, sample_size: int) -> np.ndarray:
	return np.random.choice(population, size=sample_size, replace=False)


def stratified_select(population: np.ndarray, sample_size: int, strata_count: int = 5) -> np.ndarray:
	strata_indices = np.array_split(np.argsort(population), strata_count)
	base_size = sample_size // strata_count
	remainder = sample_size % strata_count

	selected_values = []
	for stratum_index, indices in enumerate(strata_indices):
		current_size = base_size + (1 if stratum_index < remainder else 0)
		if current_size == 0:
			continue
		chosen_indices = np.random.choice(indices, size=current_size, replace=False)
		selected_values.append(population[chosen_indices])

	return np.concatenate(selected_values)


def systematic_select(population: np.ndarray, sample_size: int) -> np.ndarray:
	population_size = len(population)
	step = max(1, population_size // sample_size)
	start = np.random.randint(0, step)
	indices = start + np.arange(sample_size) * step
	return population[indices]

def mean(data: np.ndarray) -> float:
    return np.mean(data)

def standard_deviation(data: np.ndarray) -> float:
    return np.std(data, ddof=0)

def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    n = len(data)
    mean_value = mean(data)
    std_dev = standard_deviation(data)
    z_score = 1.96 
    margin_of_error = z_score * (std_dev / np.sqrt(n))
    return (mean_value - margin_of_error, mean_value + margin_of_error)

def Zadanie1() -> None:
	population = generate_population(size=100_000, mean=50.0, std=10.0)
	samples = [random_select(population, sample_size) for sample_size in SAMPLE_SIZES]
	for sample in samples:
		print(f"Sample Size: {len(sample)}")
		print(f"Mean: {mean(sample):.2f}")
		print(f"Standard Deviation: {standard_deviation(sample):.2f}")
		print("-" * 40)

def Zadanie2() -> None:
	population = generate_population(size=100_000, mean=10.0, std=15.0)
	samples = [random_select(population, sample_size) for sample_size in SAMPLE_SIZES]
	for sample in samples:
		lower, upper = confidence_interval(sample)
		print(f"Sample Size: {len(sample)}")
		print(f"Confidence Interval: ({lower:.4f}, {upper:.4f})")
		print("-" * 40)

def Zadanie3() -> None:
	population_size = 100_000
	left_size = population_size // 2
	right_size = population_size - left_size

	left_subset = generate_population(size=left_size, mean=30.0, std=5.0)
	right_subset = generate_population(size=right_size, mean=70.0, std=5.0)
	population = np.concatenate([left_subset, right_subset])

	population_mean = mean(population)
	population_std = standard_deviation(population)

	print("Rzeczywiste wartosci populacyjne (bimodalna):")
	print(f"Mean: {population_mean:.4f}")
	print(f"Standard Deviation: {population_std:.4f}")
	print("-" * 40)

	random_samples = [random_select(population, sample_size) for sample_size in SAMPLE_SIZES]
	non_random_samples = [random_select(left_subset, sample_size) for sample_size in SAMPLE_SIZES]

	print("Probki losowe:")
	for sample in random_samples:
		print(f"Sample Size: {len(sample)}")
		print(f"Mean: {mean(sample):.4f}")
		print(f"Standard Deviation: {standard_deviation(sample):.4f}")
		print("-" * 40)

	print("Probki nielosowe (tylko z jednego podzbioru):")
	for sample in non_random_samples:
		print(f"Sample Size: {len(sample)}")
		print(f"Mean: {mean(sample):.4f}")
		print(f"Standard Deviation: {standard_deviation(sample):.4f}")
		print("-" * 40)

def Zadanie4() -> None:
	population = generate_population(size=100_000, mean=50.0, std=10.0)

	print(f"Population Size: {len(population)}")
	print(f"Population Mean: {mean(population):.4f}")
	print(f"Population Standard Deviation: {standard_deviation(population):.4f}")
	print("-" * 40)

	random_samples = [random_select(population, sample_size) for sample_size in SAMPLE_SIZES]
	stratified_samples = [stratified_select(population, sample_size) for sample_size in SAMPLE_SIZES]
	systematic_samples = [systematic_select(population, sample_size) for sample_size in SAMPLE_SIZES]

	print("b) Proste losowanie (`numpy.random.choice`) :")
	for sample in random_samples:
		print(f"Sample Size: {len(sample)}")
		print(f"Mean: {mean(sample):.4f}")
		print(f"Standard Deviation: {standard_deviation(sample):.4f}")
		print("-" * 40)

	print("c) Losowanie warstwowe:")
	for sample in stratified_samples:
		print(f"Sample Size: {len(sample)}")
		print(f"Mean: {mean(sample):.4f}")
		print(f"Standard Deviation: {standard_deviation(sample):.4f}")
		print("-" * 40)

	print("d) Losowanie systematyczne:")
	for sample in systematic_samples:
		print(f"Sample Size: {len(sample)}")
		print(f"Mean: {mean(sample):.4f}")
		print(f"Standard Deviation: {standard_deviation(sample):.4f}")
		print("-" * 40)

def main() -> None:
    print("Zadanie 1:")
    Zadanie1()
    print("Zadanie 2:")
    Zadanie2()
    print("Zadanie 3:")
    Zadanie3()
    print("Zadanie 4:")
    Zadanie4()

if __name__ == "__main__":
	main()
