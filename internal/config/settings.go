package config

import (
	"encoding/csv"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
	"libs.altipla.consulting/errors"
)

var Features, Labels, TestFeatures, TestLabels *mat.Dense

func ParseSettings() error {
	f, err := os.Open("files/penguins_train.csv")
	if err != nil {
		return errors.Trace(err)
	}
	defer f.Close()

	Features, Labels, err = loadDataFromCSV(f)
	if err != nil {
		return errors.Trace(err)
	}

	f, err = os.Open("files/penguins_test.csv")
	if err != nil {
		return errors.Trace(err)
	}
	defer f.Close()

	TestFeatures, TestLabels, err = loadDataFromCSV(f)
	if err != nil {
		return errors.Trace(err)
	}

	return nil
}

func loadDataFromCSV(file *os.File) (*mat.Dense, *mat.Dense, error) {
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = 7

	rawData, err := reader.ReadAll()
	if err != nil {
		return nil, nil, errors.Trace(err)
	}

	featuresData := make([]float64, 4*len(rawData))
	labelsData := make([]float64, 3*len(rawData))

	var featuresIndex, labelsIndex int32
	for i, record := range rawData {
		if i == 0 {
			continue
		}

		for j, entry := range record {
			parsed, err := strconv.ParseFloat(entry, 64)
			if err != nil {
				return nil, nil, errors.Trace(err)
			}

			if j >= 4 && j <= 6 {
				labelsData[labelsIndex] = parsed
				labelsIndex++
				continue
			}

			featuresData[featuresIndex] = parsed
			featuresIndex++
		}
	}

	features := mat.NewDense(len(rawData), 4, featuresData)
	labels := mat.NewDense(len(rawData), 3, labelsData)

	return features, labels, nil
}
