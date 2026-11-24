import math
import re
clinicalPreProcessingDataset = [
    "Patient diagnosed with HTN which means hypertension.",
    "Hypertension (HTN) is diagnosed.",
    "Patient has HTN and high blood pressure.",
    "The patient has PNA, meaning pneumonia.",
    "Pneumonia (PNA) is diagnosed.",
    "The patient had abx (antibiotics).",
    "Patient had antibiotics.",
    "Patient had CHF.",
    "Patient has CKD.",
]
ClinicalNoteDataset = [
    "Patient has diagnosis of HTN and PNA.",
    "Patient had CHF and has CKD.",
    "The patient has abx for diabetes.",
]
medLingoCorrectDataset = {
    "HTN": "hypertension",
    "PNA": "pneumonia",
    "abx": "antibiotics",
    "CHF": "congestive heart failure",
    "CKD": "chronic kidney disease",
}
commonWords = {"with", "is", "for", "of", "and", "the", "has", "had"}
def preprocessClinicalDataset(clinicalDocuments, commonWords):
    upperCaseRegularExpressionPattern = r"\b[A-Z]{2,5}\b"
    lowerCaseRegularExpressionPattern = r"\b[a-z]{2,5}\b"
    clinicalTermsInDocuments = []
    for note in clinicalDocuments:
        upperCaseRegularExpressionPatternTokens = re.findall(upperCaseRegularExpressionPattern, note)
        lowerCaseRegularExpressionPatternTokens = re.findall(lowerCaseRegularExpressionPattern, note)
        for token in upperCaseRegularExpressionPatternTokens + lowerCaseRegularExpressionPatternTokens:
            if (
                token.lower() not in commonWords
                and token not in clinicalTermsInDocuments
            ):
                clinicalTermsInDocuments.append(token)
    return clinicalTermsInDocuments
class ClinicalLanguageModelSimulator:
    def __init__(self, minimumTermsToLearnClinicalTerm=2):
        self.minimumTermsToLearnClinicalTerm = minimumTermsToLearnClinicalTerm
        self.modelLearnedClinicalTerms = {}
        self.clinicalDocumentTermMentionedCount = {}
    def ClinicalPreProcessingDatasetClinicalTermsLearning(
        self,
        clinicalPreProcessingDocumentsDataset,
        clinicalDocumentsTermsDictionary,
        clinicalTermAndClinicalDefinitionOccuranceCalculation=True,
        clinicalTermContextCountCalculation=True,
    ):
        for eachClinicalTerm, clinicalTermDefinition in clinicalDocumentsTermsDictionary.items():
            clinicalTermAndClinicalDefinitionOccuranceCount = 0
            clinicalTermContextCount = 0
            for document in clinicalPreProcessingDocumentsDataset:
                clinicalDocumentComponent = document.lower()
                clinicalTermPresentInDocuments = eachClinicalTerm.lower() in clinicalDocumentComponent
                clinicalTermClinicalTermDefinitionPresentInDocumnets = clinicalTermDefinition.lower() in clinicalDocumentComponent
                if clinicalTermPresentInDocuments and clinicalTermClinicalTermDefinitionPresentInDocumnets:
                    clinicalTermAndClinicalDefinitionOccuranceCount += 1
                elif clinicalTermPresentInDocuments and not clinicalTermClinicalTermDefinitionPresentInDocumnets:
                    clinicalTermContextCount += 1
            clinicalTermsOccurancesCountCalculation = 0
            if clinicalTermAndClinicalDefinitionOccuranceCalculation:
                clinicalTermsOccurancesCountCalculation += clinicalTermAndClinicalDefinitionOccuranceCount
            if clinicalTermContextCountCalculation:
                clinicalTermsOccurancesCountCalculation += clinicalTermContextCount
            self.clinicalDocumentTermMentionedCount[eachClinicalTerm] = clinicalTermsOccurancesCountCalculation
            if clinicalTermsOccurancesCountCalculation >= self.minimumTermsToLearnClinicalTerm:
                self.modelLearnedClinicalTerms[eachClinicalTerm] = clinicalTermDefinition
            else:
                self.modelLearnedClinicalTerms[eachClinicalTerm] = "Clinical term is unknown. "
    def predictClinicalTermClinicalTermDefinition(self, eachClinicalTerm, oneOccuranceClinicalTerm=True):
        prediction = self.modelLearnedClinicalTerms.get(
            eachClinicalTerm, "Clinical term is unknown."
        )
        if not oneOccuranceClinicalTerm and prediction != "Clinical term is unknown.":
            if self.clinicalDocumentTermMentionedCount.get(eachClinicalTerm, 0) <= 2:
                return "Clinical term is unknown. "
        return prediction
def calculateCorrelationCalculation(variableXList, variableYList):
    countOfXVariable = len(variableXList)
    if countOfXVariable < 2:
        return 0.0
    def calculateRankNumber(values):
        sortedVariablesIndex = sorted(range(countOfXVariable), key=lambda i: values[i])
        variableRanks = [0.0] * countOfXVariable
        for variableRank, variableIndex in enumerate(sortedVariablesIndex):
            variableRanks[variableIndex] = float(variableRank + 1)
        return variableRanks
    rankOfXVariable = calculateRankNumber(variableXList)
    rankOfYVariable = calculateRankNumber(variableYList)
    meanOfXVariable = sum(rankOfXVariable) / countOfXVariable
    meanOfYVariable = sum(rankOfYVariable) / countOfXVariable
    numerator = sum(
        (rankOfXVariable[i] - meanOfXVariable) * (rankOfYVariable[i] - meanOfYVariable) for i in range(countOfXVariable)
    )
    denominatorOfXVariable = math.sqrt(sum((rankOfXVariable[i] - meanOfXVariable) ** 2 for i in range(countOfXVariable)))
    denominatorOfYVariable = math.sqrt(sum((rankOfYVariable[i] - meanOfYVariable) ** 2 for i in range(countOfXVariable)))
    if denominatorOfXVariable == 0 or denominatorOfYVariable == 0:
        return 0.0
    return numerator / (denominatorOfXVariable * denominatorOfYVariable)

def clinicalTermModelEvaluation(
    model, clinicalTermEvaluationList, correctClinicalTermDictionary, oneOccuranceClinicalTerm=True
):
    correctClinicalTermPrediction = 0
    clincialTermFrequency = []
    clinicalTermAccuracy = []
    for eachClinicalTerm in clinicalTermEvaluationList:
        if eachClinicalTerm not in correctClinicalTermDictionary:
            continue
        clinicalTermCorrectClinicalTermDefinition = correctClinicalTermDictionary[eachClinicalTerm]
        clinicalTermPredictiedClinicalTermDefinition = model.predictClinicalTermClinicalTermDefinition(
            eachClinicalTerm, oneOccuranceClinicalTerm=oneOccuranceClinicalTerm
        )
        clinicalTermCorrectness = (
            1 if clinicalTermPredictiedClinicalTermDefinition.lower() == clinicalTermCorrectClinicalTermDefinition.lower() else 0
        )
        correctClinicalTermPrediction += clinicalTermCorrectness
        clincialTermFrequency.append(model.clinicalDocumentTermMentionedCount.get(eachClinicalTerm, 0))
        clinicalTermAccuracy.append(clinicalTermCorrectness)
    totalClinicalTermsEvaluated = len(clinicalTermAccuracy)
    accuracyScore = (
        correctClinicalTermPrediction / totalClinicalTermsEvaluated if totalClinicalTermsEvaluated > 0 else 0.0
    )
    correlationVariable = calculateCorrelationCalculation(clincialTermFrequency, clinicalTermAccuracy)
    return accuracyScore, correlationVariable
if __name__ == "__main__":
    print("--- PREPROCESSING CLINICAL DATASTETS ---")
    clinicalTermsProcessing = preprocessClinicalDataset(
        ClinicalNoteDataset, commonWords
    )
    medLingoClinicalTerm = [
        term for term in clinicalTermsProcessing if term in medLingoCorrectDataset
    ]
    print("MedLingo Dataset Terms:", medLingoClinicalTerm)
    print("\n--- FULL PROJECT TESTING ---")
    fullLargeLanguageModel = ClinicalLanguageModelSimulator(minimumTermsToLearnClinicalTerm=2)
    fullLargeLanguageModel.ClinicalPreProcessingDatasetClinicalTermsLearning(
        clinicalPreProcessingDataset,
        medLingoCorrectDataset,
        clinicalTermAndClinicalDefinitionOccuranceCalculation=True,
        clinicalTermContextCountCalculation=True,
    )
    fullProjectAccuracy, fullProjectCorrelationScore = clinicalTermModelEvaluation(
        fullLargeLanguageModel,
        medLingoClinicalTerm,
        medLingoCorrectDataset,
        oneOccuranceClinicalTerm=True,
    )
    print(f"Full Large Languagae Model Capacity Project (All Features Implementation) -> Accuracy Score: {fullProjectAccuracy:.2f} | Correlation Score: {fullProjectCorrelationScore:.2f}")
    print("Clinical Term Count:", fullLargeLanguageModel.clinicalDocumentTermMentionedCount)
    print("\n--- ABLATION TESTING RESULTS ---")
    ablationTestingModel = ClinicalLanguageModelSimulator(minimumTermsToLearnClinicalTerm=3)
    ablationTestingModel.ClinicalPreProcessingDatasetClinicalTermsLearning(clinicalPreProcessingDataset,medLingoCorrectDataset,clinicalTermAndClinicalDefinitionOccuranceCalculation=True,clinicalTermContextCountCalculation=True)
    ablationTestingAccuracy, ablationTestingCorrelationScore = clinicalTermModelEvaluation(ablationTestingModel,medLingoClinicalTerm,medLingoCorrectDataset,oneOccuranceClinicalTerm=True)
    print(f"Small Language Model Capacity (Increased Threshold For Rare Clinical Term Comprehension) -> Accuracy Score: {ablationTestingAccuracy:.2f} | Correlation Score: {ablationTestingCorrelationScore:.2f}")