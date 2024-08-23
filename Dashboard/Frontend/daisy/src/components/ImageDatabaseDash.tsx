import React, { useEffect, useState} from 'react';
import { useRecoilState } from 'recoil';
import { DefaultService, ImageSchemaOut, OpenAPI } from '../clients/general/index';
import { DefaultService as InferenceService, OpenAPI as InferenceAPI, Body_upload_model_upload_model_post } from '../clients/inference/index';
import { imagesState , selectedModelState} from '../states/atoms/index';
import { inferenceResultsState } from '../states/atoms/inferenceResultsAtom';
import { modelsState, inferenceProgressState } from '../states/atoms/index';
import { Button, Container, Typography, Box, TextField } from '@mui/material';
import { LinearProgress, Dialog } from '@mui/material';

import { CloudUpload, Send, Delete } from '@mui/icons-material';
import Papa from 'papaparse';

// Set the base URL for the API
// OpenAPI.BASE = 'http://localhost:8080';
// OpenAPI.BASE = 'http://51.20.66.106:8080';
OpenAPI.BASE = 'https://155e8438ecc5.ngrok.app'

InferenceAPI.BASE = 'https://6e81cdfc5c99.ngrok.app';
// InferenceAPI.BASE = 'http://51.20.66.106:8081';
// InferenceAPI.BASE = 'http://localhost:8081';


const ImageDatabaseDash = () => {
  const [images, setImages] = useRecoilState(imagesState);
  const [models, setModels] = useRecoilState(modelsState);
  const [inferenceResults, setInferenceResults] = useRecoilState(inferenceResultsState);
  const [isResultsVisible, setResultsVisible] = useState(false);
  const [inferenceProgress, setInferenceProgress] = useRecoilState(inferenceProgressState);


  const [zipFile, setZipFile] = useState<File | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState<string>('');

  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const [selectedModelAtom, setSelectedModelAtom] = useRecoilState(selectedModelState);

  useEffect(() => {
    handleGetAllImages();
  }, []);

  useEffect(() => {
    const fetchAllData = () => {
      handleGetAllImages();
      handleGetAllModels();
      handleGetAllInferenceResults();
      handleGetProgressState();
    };

    fetchAllData(); // Fetch initial data

    const interval = setInterval(fetchAllData, 2000); // Fetch data every second

    return () => clearInterval(interval); // Cleanup interval on component unmount
  }, []);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>, setFile: React.Dispatch<React.SetStateAction<File | null>>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
    }
  };


  // const progressBarU

  const handleSendCsvData = () => {
    if (!csvFile) return;

    setIsProcessing(true);

    Papa.parse(csvFile, {
      header: true,
      complete: async (results: any) => {
        const data = results.data as { img_url: string, lat: string, long: string, time: string, img_field: string}[];
        
        for (const record of data) {
          try {
            await DefaultService.createImageInsertImgPost({
              img_url: record.img_url,
              img_field: record.img_field,
              coordinates: [parseFloat(record.lat), parseFloat(record.long)],
              time: parseFloat(record.time),
            });
          } catch (error) {
            console.error('Error creating image record:', error);
          }
        }

        setIsProcessing(false);
        alert('Data upload complete!');
        handleGetAllImages(); // Refresh images list
      },
      error: (error: any) => {
        console.error('Error parsing CSV:', error);
        setIsProcessing(false);
      }
    });
  };

  const handleSendFiles = async () => {
    if (!zipFile) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', zipFile);

    try {
      const response = await fetch(`${OpenAPI.BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        alert('ZIP file uploaded and processed successfully!');
      } else {
        console.error('Failed to upload ZIP file:', response.statusText);
      }
    } catch (error) {
      console.error('Error uploading ZIP file:', error);
    } finally {
      setIsProcessing(false);
    }
  };


  const handleSendAll = async () => {

    if (!zipFile || !csvFile ) {
      alert("Upload both the zip file and the csv file!")
      return;
    }
    handleSendFiles();
    handleSendCsvData();

  }

  const handleUploadModel = async () => {
    if (!modelFile || !modelName) {
      alert("Please provide a model file and name.");
      return;
    }

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', modelFile);

    const body: Body_upload_model_upload_model_post = {
      file: modelFile,
    };

    try {
      const response = await InferenceService.uploadModelUploadModelPost(body, modelName);

      if (response) {
        alert('Model file uploaded successfully!');
      }

    } catch (error) {
      console.error('Error uploading model file:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleGetAllImages = async () => {
    try {
      const response = await DefaultService.getAllImagesGetAllImagesGet();
      setImages(response);
      console.log('GET All response:', response);
    } catch (error) {
      console.error('Error fetching all images:', error);
    }
  };

  const handleGetProgressState = async () => {
    try {
      const response = await DefaultService.getProgressGetInferenceProgressGet();
      console.log('GET Progress Inference response:', response);

      setInferenceProgress(response);
      
    } catch (error) {
      console.error('Error fetching progress :', error);
    }
  }

  const handleGetAllModels = async () => {
    try {
      const response = await InferenceService.getAllModelsModelsGet();
      setModels(response);
      console.log('GET All Models response:', response);
      
    } catch (error) {
      console.error('Error fetching all models:', error);
    }
  }

  const handleGetAllInferenceResults = async () => {
    try {
      const response = await InferenceService.getAllInferenceResultsInferenceResultsGet();
      setInferenceResults(response);
      console.log('GET All Inference Results response:', response);
    } catch (error) {
      console.error('Error fetching all inference results', error);
    }
  }

  const handleDeleteImage = async (imageId: string) => {
    try {
      await DefaultService.deleteImageDeleteImgImageIdDelete(imageId);
      alert(`Image ${imageId} deleted successfully!`);
      handleGetAllImages(); // Refresh images list
    } catch (error) {
      console.error(`Error deleting image ${imageId}:`, error);
    }
  };

  const handleDeleteAllImages = async () => {
    try {
      await DefaultService.deleteAllImagesDeleteAllImagesDelete();
      alert('All images deleted successfully!');
      handleGetAllImages(); // Refresh images list
    } catch (error) {
      console.error('Error deleting all images:', error);
    }
  };

  const handleDeleteAllPredictions = async () => {
    try {
      await InferenceService.deleteAllPredictionsDeleteAllPredictionsDelete();
      alert('All predictions deleted successfully!');
    } catch (error) {
      console.error('Error deleting all images:', error);
    }
  };

  const handleImageClick = async (imageId: string) => {
    try {
      const response = await DefaultService.getImageFileGetImageFileImageIdGet(imageId);
      if (!response || !response.image) {
        throw new Error('No image data received');
      }

      setSelectedImage(`data:image/jpeg;base64,${response.image}`);
    } catch (error) {
      console.error('Error fetching image file:', error);
    }
  };

  const handleModelSelection = async (modelId: string) => {
    try {

      setSelectedModel(modelId);
      setSelectedModelAtom(modelId);
      console.log('Selected Model:', modelId);
    } catch (error) {
      console.error('Error setting the model', error);
    }
  }

  const handlePredictImage = async (imageUrl: string, imageId: string) => {
    try {
      if (!selectedModel) {
        alert('Select a model to use for prediction!');
        throw new Error('No model selected');
      } else {
        const response = await InferenceService.predictPredictImgUrlPost(
          imageUrl, imageId, selectedModel);
  
        console.log('Prediction response:', response);
      }
    } catch (error) {
      console.error('Error predicting image:', error);
    }
  }
  
  const handlePredictAllImages = async () => {
    try {
      if (!selectedModel) {
        alert('Select a model to use for prediction!');
        throw new Error('No model selected');
      } else {
        const response = await InferenceService.predictAllPredictAllPost();
        console.log('Prediction Response:', response);
      }
      
    } catch (error) {
      console.error('Error predicting images:', error);
    }
  }

  const groupedResults = inferenceResults.reduce((acc:any, result: any) => {
    if (!acc[result.img_id]) {
      acc[result.img_id] = [];
    }
    acc[result.img_id].push(result);
    return acc;
  }, {});


  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Image Database Dashboard
      </Typography>

    <Dialog
      open={inferenceProgress.currently_training}
      PaperProps={{
        style: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          padding: '20px',
          borderRadius: '10px',
          textAlign: 'center',
        },
      }}
    >
      <Box sx={{ width: '300px' }}>
        <Typography variant="h6" color="white" gutterBottom>
          Predicting images
        </Typography>
        <LinearProgress
          variant="determinate"
          value={inferenceProgress.percentage_completed} 
          sx={{
            height: '10px',
            borderRadius: '5px',
            marginBottom: '10px',
            backgroundColor: 'rgba(255, 255, 255, 0.2)',
          }}
        />
        <Typography variant="body1" color="white">
          {Math.round(inferenceProgress.percentage_completed)}%
        </Typography>
      </Box>
    </Dialog>


    <Box display="flex" justifyContent="flex-start" marginBottom={2}>

    <Box sx={{ width: '80vh' }}>


      <Box display="flex" alignItems="center" marginBottom={2}>
          <input
            accept=".zip"
            id="upload-zip"
            type="file"
            style={{ display: 'none' }}
            onChange={(e) => handleFileUpload(e, setZipFile)}
          />
          <label htmlFor="upload-zip">
            <Button
              variant="contained"
              color="primary"
              startIcon={<CloudUpload />}
              component="span"
              style={{ marginRight: 10 }}
            >
              Select ZIP
            </Button>
          </label>
          {zipFile && <Typography variant="body1">{zipFile.name}</Typography>}
        </Box>
        <Box display="flex" alignItems="center" marginBottom={2}>
          <input
            accept=".csv"
            id="upload-csv"
            type="file"
            style={{ display: 'none' }}
            onChange={(e) => handleFileUpload(e, setCsvFile)}
          />
          <label htmlFor="upload-csv">
            <Button
              variant="contained"
              color="secondary"
              startIcon={<CloudUpload />}
              component="span"
              style={{ marginRight: 10 }}
            >
              Select CSV
            </Button>
          </label>
          {csvFile && <Typography variant="body1">{csvFile.name}</Typography>}
        </Box>
        <Box display="flex" alignItems="center" marginBottom={2}>
          <input
            accept=".zip"
            id="upload-model"
            type="file"
            style={{ display: 'none' }}
            onChange={(e) => handleFileUpload(e, setModelFile)}
          />
          <label htmlFor="upload-model">
            <Button
              variant="contained"
              color="secondary"
              startIcon={<CloudUpload />}
              component="span"
              style={{ marginRight: 10 }}
            >
              Upload Model
            </Button>
          </label>
          {modelFile && <Typography variant="body1">{modelFile.name}</Typography>}
        </Box>
        <Box display="flex" justifyContent="flex-start" marginBottom={2}> 
          <TextField
            label="Model Name"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            fullWidth
            margin="normal"
            variant="outlined"
          />

        </Box >

    </Box>

      {selectedImage && (
            <Box marginLeft={30}>
              <Typography variant="h6">Selected Image</Typography>
              <img src={selectedImage} alt="Selected" style={{ maxWidth: '40%' }} />
            </Box>
          )}
    </Box>

      <Box display="flex" justifyContent="flex-start" marginBottom={2}>
        {/* Used for Debugging Purposes */}
        {/* <Button
          variant="contained"
          color="inherit"
          startIcon={<Send />}
          onClick={handleSendCsvData}
          disabled={isProcessing}
        >
          {isProcessing ? 'Processing...' : 'Send CSV'}
        </Button> */}
        {/* <Button
          variant="contained"
          color="primary"
          startIcon={<Send />}
          onClick={handleSendFiles}
          disabled={isProcessing || !zipFile}
          style={{ marginLeft: 10 }}
        >
          {isProcessing ? 'Processing...' : 'Send ZIP'}
        </Button> */}
        <Button
          variant="contained"
          color="primary"
          startIcon={<Send />}
          onClick={() => {
            handleSendAll();
          }}
          disabled={isProcessing || (!zipFile || !csvFile)}
          style={{ marginLeft: 30 }}
        >
          {isProcessing ? 'Processing...' : 'Upload Data'}
        </Button>
        <Button
          variant="contained"
          color="primary"
          startIcon={<CloudUpload />}
          onClick={handleUploadModel}
          disabled={isProcessing || !modelFile}
          style={{ marginLeft: 10 }}
        >
          {isProcessing ? 'Processing...' : 'Upload Model'}
        </Button>
      </Box>

      <Box display="flex" justifyContent="flex-start" marginBottom={2}>
        {/* Used for Debugging Purposes */}
        {/* <Button 
          variant="contained" 
          color="primary" 
          onClick={() => {
            handleGetAllImages();
            handleGetAllModels();
            handleGetAllInferenceResults();
          }} 
          style={{ marginRight: 10 }}
        >
          Get All Images, Models, Results
        </Button> */}
        <Button
          variant="contained"
          color="error"
          onClick={handleDeleteAllImages}
          style={{ marginLeft: 30 }}
        >
          Delete All Images
        </Button>
        <Button
          variant="contained"
          color="error"
          onClick={handleDeleteAllPredictions}
          style={{ marginLeft: 30 }}
        >
          Delete All Predictions
        </Button>
        <Button
          variant="contained"
          color="success"
          onClick={() => handlePredictAllImages()}
          style={{ marginLeft: 30 }}
        >
          Predict All Images
        </Button>
      </Box>




{models.length > 0 && (
        <Box marginTop={3}>
          <Typography variant="h6">Models List</Typography>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Model Path</th>
                <th>Model Name</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model, index) => (
                <tr key={index}>
                  <td>{model.id}</td>
                  <td>{model.model_path}</td>
                  <td>{model.model_name}</td>
                  <td>
                    <Button
                      variant="contained"
                      color="secondary"
                      startIcon={<Delete />}
                      onClick={() => handleDeleteImage(model.id)}
                    >
                      Delete
                    </Button>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={() => handleModelSelection(model.model_path)}
                    >
                      Select Model
                    </Button>

                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Box>
      )}

    
      {images.length > 0 && (
        <Box marginTop={3}>
          <Typography variant="h6">Images List</Typography>
          <table style={{ width: '100%', marginTop: '10px', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th>ID</th>
                <th>Image URL</th>
                <th>Coordinates</th>
                <th>Time</th>
                <th>Created At</th>
                <th>Updated At</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {images.map((image, index) => (
                <React.Fragment key={index}>
                <tr>
                <td style={{ border: '5px solid #ddd', padding: '8px' }} ><strong>{image.id}</strong></td>
                <td style={{ border: '5px solid #ddd', padding: '8px' }} ><strong>{image.img_url}</strong></td>
                <td style={{ border: '5px solid #ddd', padding: '8px' }}><strong>{image.coordinates.join(', ')}</strong></td>
                <td style={{ border: '5px solid #ddd', padding: '8px' }}><strong>{image.time}</strong></td>
                <td style={{ border: '5px solid #ddd', padding: '8px' }}><strong>{new Date(image.created_at * 1000).toLocaleString()}</strong></td>
                <td style={{ border: '5px solid #ddd', padding: '8px' }}><strong>{new Date(image.updated_at * 1000).toLocaleString()}</strong></td>
                <td>
                    <Button
                      variant="contained"
                      color="secondary"
                      startIcon={<Delete />}
                      onClick={() => handleDeleteImage(image.id)}
                    >
                      Delete
                    </Button>

                    <Button
                      variant="contained"
                      color="primary"
                      onClick={() => handleImageClick(image.id)}
                    >
                      View Image
                    </Button>

                    <Button
                      variant="contained"
                      color="success"
                      onClick={() => handlePredictImage(image.img_url, image.id)}
                    >
                      Predict
                    </Button>

                    {groupedResults[image.img_url] && groupedResults[image.img_url].length > 0 && (
                    <Button
                      variant="contained"
                      color="inherit"
                      onClick={() => setResultsVisible(!isResultsVisible)}
                    >
                      {isResultsVisible ? 'Hide Results' : 'Show Results'}
                    </Button>
                  )}
                  </td>
                </tr>

                  {isResultsVisible && groupedResults[image.img_url] && groupedResults[image.img_url].length > 0 && (
                    <tr>
                      <td colSpan={7}>
                        <Box marginTop={2}>
                          <Typography variant="subtitle1">Inference Results:</Typography>
                          <table style={{ width: '100%', marginTop: '10px', borderCollapse: 'collapse' }}>
                            <thead>
                              <tr>
                                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Model ID</th>
                                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Labels</th>
                                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Results</th>
                                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Binary Results</th>

                              </tr>
                            </thead>
                            <tbody>
                              {groupedResults[image.img_url].map((result: any, idx: any) => (
                                <tr key={idx}>
                                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{result.model_id}</td>
                                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{result.labels.join(', ')}</td>
                                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{result.results.join(', ')}</td>
                                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{result.binary_results.join(', ')}</td>

                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </Box>
                      </td>
                    </tr>
                  )}
                  </React.Fragment>
              ))}
            </tbody>
          </table>
        </Box>
      )}

    </Container>
  );
};

export default ImageDatabaseDash;
