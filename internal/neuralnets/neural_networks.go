package neuralnets

import (
	"math"
	"math/rand"

	log "github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"libs.altipla.consulting/errors"

	"code.ia/internal/config"
)

type NeuralNetwork struct {
	Settings      NeuralNetworkSettings
	WeightsHidden *mat.Dense
	BiasesHidden  *mat.Dense
	WeightsOut    *mat.Dense
	BiasesOut     *mat.Dense
}

// Se utiliza la función sigmoide como función de activación.
func (network *NeuralNetwork) Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// La derivada de la función sigmoide. Es necesaria para la retropropagación.
func (network *NeuralNetwork) SigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

func (network *NeuralNetwork) Print() {
	log.Infof("\nPesos de la capa oculta = \n %v\n\n", mat.Formatted(network.WeightsHidden, mat.Prefix(" ")))
	log.Infof("\nBiases de la capa oculta = \n %v\n\n", mat.Formatted(network.BiasesHidden, mat.Prefix(" ")))
	log.Infof("\nPesos de la capa de salida = \n %v\n\n", mat.Formatted(network.WeightsOut, mat.Prefix(" ")))
	log.Infof("\nBiases de la capa de salida = \n %v\n\n", mat.Formatted(network.BiasesOut, mat.Prefix(" ")))
}

// Configuración de la red neuronal.
type NeuralNetworkSettings struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	Epochs        int
	LearningRate  float64
}

func NewNetwork(settings NeuralNetworkSettings) *NeuralNetwork {
	return &NeuralNetwork{Settings: settings}
}

// Trains the neural network using backpropagation.
func (network *NeuralNetwork) Train(features, labels *mat.Dense) error {
	// Primer paso. Inicializamos los pesos y desplazamientos de la red con valores aleatorios.
	dataWeightsHidden := make([]float64, network.Settings.HiddenNeurons*network.Settings.InputNeurons)
	dataBiasesHidden := make([]float64, network.Settings.HiddenNeurons)
	dataWeightsOut := make([]float64, network.Settings.OutputNeurons*network.Settings.HiddenNeurons)
	dataBiasesOut := make([]float64, network.Settings.OutputNeurons)
	for _, values := range [][]float64{dataWeightsHidden, dataBiasesHidden, dataWeightsOut, dataBiasesOut} {
		for i := range values {
			values[i] = rand.Float64()
		}
	}

	weightsHidden := mat.NewDense(network.Settings.InputNeurons, network.Settings.HiddenNeurons, dataWeightsHidden)
	biasesHidden := mat.NewDense(1, network.Settings.HiddenNeurons, dataBiasesHidden)
	weightsOut := mat.NewDense(network.Settings.HiddenNeurons, network.Settings.OutputNeurons, dataWeightsOut)
	biasesOut := mat.NewDense(1, network.Settings.OutputNeurons, dataBiasesOut)

	output := new(mat.Dense)
	// Número de ciclos.
	for i := 0; i < network.Settings.Epochs; i++ {
		// Paso 1. Feedforward.
		// Capa oculta.
		// z^l = w^l*a^l-1+b^l
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(features, weightsHidden)
		addBiasesHidden := func(_, col int, v float64) float64 {
			return v + biasesHidden.At(0, col)
		}
		hiddenLayerInput.Apply(addBiasesHidden, hiddenLayerInput)

		// a^l=sigmoide(z)
		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 {
			return network.Sigmoid(v)
		}
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		// Capa de salida.
		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, weightsOut)
		addBiasesOut := func(_, col int, v float64) float64 {
			return v + biasesOut.At(0, col)
		}
		outputLayerInput.Apply(addBiasesOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Paso 2. Calcular el error.
		// Resultados esperados - resultados predichos.
		networkError := new(mat.Dense)
		networkError.Sub(labels, output)

		// Paso 3. Propagar los errores hacia atrás.
		// Derivada de la activación en la capa de salida.
		spOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 {
			return network.SigmoidPrime(v)
		}
		spOutputLayer.Apply(applySigmoidPrime, output)

		// Derivada de la activación en la capa de oculta.
		spHiddenLayer := new(mat.Dense)
		spHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		// Error en la capa de salida.
		deltasOutput := new(mat.Dense)
		deltasOutput.MulElem(networkError, spOutputLayer)

		errorsHiddenLayer := new(mat.Dense)
		errorsHiddenLayer.Mul(deltasOutput, weightsOut.T())

		// Error en la capa oculta.
		deltasHiddenLayer := new(mat.Dense)
		deltasHiddenLayer.MulElem(errorsHiddenLayer, spHiddenLayer)

		// Paso 4. Ajustar pesos y desplazamientos.
		// Actualizar los pesos de la capa de salida.
		nablasWeightsOut := new(mat.Dense)
		nablasWeightsOut.Mul(hiddenLayerActivations.T(), deltasOutput)
		nablasWeightsOut.Scale(network.Settings.LearningRate, nablasWeightsOut)
		weightsOut.Add(weightsOut, nablasWeightsOut)

		// Actualizar los desplazamientos de la capa de salida
		nablasBiasesOut, err := sumCols(deltasOutput)
		if err != nil {
			return errors.Trace(err)
		}
		// Se escalan los ajustes con la tasa de aprendizaje.
		nablasBiasesOut.Scale(network.Settings.LearningRate, nablasBiasesOut)
		biasesOut.Add(biasesOut, nablasBiasesOut)

		// Actualizar los pesos de la capa de oculta.
		nablasWeightsHidden := new(mat.Dense)
		nablasWeightsHidden.Mul(features.T(), deltasHiddenLayer)
		nablasWeightsHidden.Scale(network.Settings.LearningRate, nablasWeightsHidden)
		weightsHidden.Add(weightsHidden, nablasWeightsHidden)

		// Actualizar los desplazamientos de la capa de oculta.
		nablasBiasesHidden, err := sumCols(deltasHiddenLayer)
		if err != nil {
			return errors.Trace(err)
		}
		nablasBiasesHidden.Scale(network.Settings.LearningRate, nablasBiasesHidden)
		biasesHidden.Add(biasesHidden, nablasBiasesHidden)
	}

	// Se actualizan los pesos y desplazamientos con los valores obtenidos.
	network.WeightsHidden = weightsHidden
	network.BiasesHidden = biasesHidden
	network.WeightsOut = weightsOut
	network.BiasesOut = biasesOut

	return nil
}

func (network *NeuralNetwork) Validate() (float64, error) {
	// Se realiza el proceso de feedforward con el conjunto de datos de prueba.
	predictions, err := network.test(config.TestFeatures)
	if err != nil {
		return -1, errors.Trace(err)
	}

	var hits, classes int
	rows, _ := predictions.Dims()
	for i := 0; i < rows; i++ {
		labelRow := mat.Row(nil, i, config.TestLabels)
		for j, label := range labelRow {
			if label == 1.0 {
				classes = j
				break
			}
		}

		if predictions.At(i, classes) == floats.Max(mat.Row(nil, i, predictions)) {
			hits++
		}
	}
	accuracy := float64(hits) / float64(rows)

	return accuracy, nil
}

// Realiza una predicción.
func (network *NeuralNetwork) test(features *mat.Dense) (*mat.Dense, error) {
	if network.WeightsHidden == nil || network.WeightsOut == nil || network.BiasesHidden == nil || network.BiasesOut == nil {
		return nil, errors.Errorf("los pesos y desplamientos deben estar inicializados")
	}

	// Sólo realiza el proceso de feedforward.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(features, network.WeightsHidden)
	addBiasesHidden := func(_, col int, v float64) float64 {
		return v + network.BiasesHidden.At(0, col)
	}
	hiddenLayerInput.Apply(addBiasesHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 {
		return network.Sigmoid(v)
	}
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, network.WeightsOut)
	addBiasesOut := func(_, col int, v float64) float64 {
		return v + network.BiasesOut.At(0, col)
	}
	outputLayerInput.Apply(addBiasesOut, outputLayerInput)

	output := new(mat.Dense)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

// Suma las columnas de una matriz dejando intacta la matriz.
func sumCols(m *mat.Dense) (*mat.Dense, error) {
	_, nCols := m.Dims()

	var output *mat.Dense
	data := make([]float64, nCols)
	for i := 0; i < nCols; i++ {
		col := mat.Col(nil, i, m)
		data[i] = floats.Sum(col)
	}
	output = mat.NewDense(1, nCols, data)

	return output, nil
}
