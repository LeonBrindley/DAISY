import React, { useEffect, useState} from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import { DefaultService, ImageSchemaOut, OpenAPI } from '../clients/general/index';
import { DefaultService as InferenceService, OpenAPI as InferenceAPI, Body_upload_model_upload_model_post } from '../clients/inference/index';
import { imagesState, selectedModelState } from '../states/atoms/index';
import { inferenceResultsState } from '../states/atoms/inferenceResultsAtom';
import { modelsState, inferenceProgressState } from '../states/atoms/index';
import { Button, Container, Typography, Box, TextField } from '@mui/material';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,Card, CardContent, CircularProgress
} from '@mui/material';
import { LinearProgress, Dialog } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { CloudUpload, Send, Delete } from '@mui/icons-material';
import Papa from 'papaparse';
import BlurOnIcon from '@mui/icons-material/BlurOn';

// Set the base URL for the API
// OpenAPI.BASE = 'http://localhost:8080';
// OpenAPI.BASE = 'http://51.20.66.106:8080';
OpenAPI.BASE = 'https://5b4f5af47258.ngrok.app'

InferenceAPI.BASE = 'https://5d5f63d26239.ngrok.app';
// InferenceAPI.BASE = 'http://51.20.66.106:8081';
// InferenceAPI.BASE = 'http://localhost:8081';


interface ImageResponse {
  image: string; // Assuming the image is returned as a Base64 string
}

const HomeView = () => {
  const [images, setImages] = useRecoilState(imagesState);
  const [models, setModels] = useRecoilState(modelsState);
  const [inferenceResults, setInferenceResults] = useRecoilState(inferenceResultsState);
  const [isResultsVisible, setResultsVisible] = useState(false);
  const [inferenceProgress, setInferenceProgress] = useRecoilState(inferenceProgressState);
  const [imageData, setImageData] = useState<{ [key: string]: string }>({});


  const [zipFile, setZipFile] = useState<File | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState<string>('');

  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  // const [selectedModel, setSelectedModel] = useState<string | null>('sensor-cdt-daisy-models/inaturalist_uf5_pFalse_lr0.0001.zip');
  const selectedModel = useRecoilValue(selectedModelState)

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


  useEffect(() => {
    const fetchImages = async () => {
      const newImageData: { [key: string]: string } = {}; // Object to store image data

      for (const image of images) {
        try {
          // Assume the response is typed as ImageResponse
          const response: ImageResponse | null = await DefaultService.getImageFileGetImageFileImageIdGet(image.id);

          if (response && response.image) {
            newImageData[image.id] = `data:image/jpeg;base64,${response.image}`;
          }
        } catch (error) {
          console.error(`Error fetching image with ID ${image.id}:`, error);
        }
      }

      // Update state with the new image data
      setImageData(newImageData);
    };

    fetchImages();
    // handlePredictAllImages();
  }, [images]);


  // const progressBarU

  const handleSendCsvData = () => {
    if (!csvFile) return;

    setIsProcessing(true);

    Papa.parse(csvFile, {
      header: true,
      complete: async (results: any) => {
        const data = results.data as { img_url: string, lat: string, long: string, time: string, img_field: string }[];
        
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

  // const handleModelSelection = async (modelId: string) => {
  //   try {

  //     setSelectedModel(modelId);
  //     console.log('Selected Model:', modelId);
  //   } catch (error) {
  //     console.error('Error setting the model', error);
  //   }
  // }

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


  const handleDrop = (event: React.DragEvent<HTMLDivElement>, setFile: React.Dispatch<React.SetStateAction<File | null>>) => {
    event.preventDefault();
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      setFile(event.dataTransfer.files[0]);
      event.dataTransfer.clearData();
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };


  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Daisy: Upload Images
      </Typography>
    
    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: '20px' }}>
    <Box sx={{ width: '50%'}}>
      {/* ZIP Upload */}
      <Box
        display="flex"
        alignItems="center"
        justifyContent="center"
        marginBottom={3}
        onDrop={(event) => handleDrop(event, setZipFile)}
        onDragOver={handleDragOver}
        sx={{
          border: '2px dashed #90caf9',
          borderRadius: '12px',
          padding: '30px',
          cursor: 'pointer',
          backgroundColor: '#f5f5f5',
          transition: 'background-color 0.3s, box-shadow 0.3s',
          '&:hover': {
            backgroundColor: '#e3f2fd',
            boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.1)',
          },
        }}
      >
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
            sx={{
              padding: '16px 32px',
              fontSize: '1.2rem',
            }}
          >
            Select Images Zip
          </Button>
        </label>
        {zipFile && (
          <Typography variant="body1" sx={{ fontWeight: 'bold', marginLeft: 2 }}>
            {zipFile.name}
          </Typography>
        )}
      </Box>

      {/* CSV Upload */}
      <Box
        display="flex"
        alignItems="center"
        justifyContent="center"
        marginBottom={3}
        onDrop={(event) => handleDrop(event, setCsvFile)}
        onDragOver={handleDragOver}
        sx={{
          border: '2px dashed #ffcc80',
          borderRadius: '12px',
          padding: '30px',
          cursor: 'pointer',
          backgroundColor: '#f5f5f5',
          transition: 'background-color 0.3s, box-shadow 0.3s',
          '&:hover': {
            backgroundColor: '#fff3e0',
            boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.1)',
          },
        }}
      >
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
            sx={{
              padding: '16px 32px',
              fontSize: '1.2rem',
            }}
          >
            Select Images CSV
          </Button>
        </label>
        {csvFile && (
          <Typography variant="body1" sx={{ fontWeight: 'bold', marginLeft: 2 }}>
            {csvFile.name}
          </Typography>
        )}
      </Box>

      </Box>

      {/* Predicting Images Box */}
    {inferenceProgress.currently_training && (
      <Box
        sx={{
          width: '20%',
          padding: 3,
          backgroundColor: '#9BC53D',
          borderRadius: '12px',
          boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.1)',
          textAlign: 'center',
        }}
      >
        <Typography variant="h6" color="white" gutterBottom>
          Predicting Images
        </Typography>
        <LinearProgress
          variant="determinate"
          value={inferenceProgress.percentage_completed}
          sx={{
            height: '12px',
            borderRadius: '6px',
            marginBottom: '16px',
            backgroundColor: 'rgba(255, 255, 255, 0.3)',
          }}
        />
        <Typography variant="body1" color="white">
          {Math.round(inferenceProgress.percentage_completed)}% completed
        </Typography>
      </Box>
    )}

    </Box>

      {/* Upload Data Button */}
      <Box sx={{ width: '50%'}} display="flex" justifyContent="center" marginBottom={2} marginLeft={3}>
        <Button
          variant="contained"
          color="primary"
          startIcon={<Send />}
          onClick={() => {
            handleSendAll();
          }}
          disabled={isProcessing || (!zipFile || !csvFile)}
          sx={{
            marginRight: '16px',
            padding: '16px 32px',
            fontSize: '1.2rem',
            backgroundColor: isProcessing ? '#bdbdbd' : '#1976d2',
            '&:hover': {
              backgroundColor: isProcessing ? '#bdbdbd' : '#1565c0',
            },
          }}
        >
          {isProcessing ? 'Processing...' : 'Upload Data'}
        </Button>
        <Button
          variant="contained"
          color="primary"
          startIcon={<BlurOnIcon/>}
          onClick={() => {
            handlePredictAllImages();
          }}
          disabled={inferenceProgress.currently_training}
          sx={{
            padding: '16px 32px',
            fontSize: '1.2rem',
            backgroundColor: isProcessing ? '#bdbdbd' : '#1976d2',
            '&:hover': {
              backgroundColor: isProcessing ? '#bdbdbd' : '#1565c0',
            },
          }}
        >
          {isProcessing ? 'Processing...' : 'Analyse Data'}
        </Button>
      </Box>



{images.length > 0 && (
        <Box marginTop={3}>
          <Typography variant="h6" gutterBottom>
            Images List
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Image</strong></TableCell>
                  <TableCell><strong>Image Name</strong></TableCell>
                  <TableCell><strong>Coordinates</strong></TableCell>
                  <TableCell><strong>Actions</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {images.map((image, index) => (
                  <React.Fragment key={index}>
                    <TableRow>
                      <TableCell>
                        {imageData[image.id] ? (
                          <img
                            src={imageData[image.id]}
                            alt={`Image ${image.id}`}
                            style={{ width: '100px', height: '100px', objectFit: 'cover' }}
                          />
                        ) : (
                          <Typography variant="body2" color="textSecondary">
                            Loading...
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>{image.img_url}</TableCell>
                      <TableCell>{image.coordinates.join(', ')}</TableCell>
                      <TableCell>
                      <Box display="flex" alignItems="center">
                        <Button
                          variant="contained"
                          color="secondary"
                          startIcon={<Delete />}
                          onClick={() => handleDeleteImage(image.id)}
                          sx={{ marginRight: 1 }}
                        >
                          Delete
                        </Button>

                        {groupedResults[image.img_url] && groupedResults[image.img_url].length > 0 ? (
                          <CheckCircleIcon color="success" fontSize="large" sx={{ marginRight: 1 }} />
                        ) : (
                          <CircularProgress size={24} sx={{ marginRight: 1 }} />

                        )}

                        {groupedResults[image.img_url] && groupedResults[image.img_url].length > 0 && (
                          <Button
                            variant="contained"
                            color="inherit"
                            onClick={() => setResultsVisible(!isResultsVisible)}
                          >
                            {isResultsVisible ? 'Hide Results' : 'Show Results'}
                          </Button>
                        )}
                      </Box>
                      </TableCell>
                    </TableRow>

                    {isResultsVisible && groupedResults[image.img_url] && groupedResults[image.img_url].length > 0 && (
                      <TableRow>
                        <TableCell colSpan={4}>
                          <Box marginTop={2}>
                            <Typography variant="subtitle1">Inference Results:</Typography>
                            <Table size="small" style={{ marginTop: '10px', borderCollapse: 'collapse' }}>
                              <TableHead>
                                <TableRow>
                                  <TableCell>Model ID</TableCell>
                                  <TableCell>Labels</TableCell>
                                  <TableCell>Results</TableCell>
                                  <TableCell>Binary Results</TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {groupedResults[image.img_url].map((result: any, idx:any) => (
                                  <TableRow key={idx}>
                                    <TableCell>{result.model_id}</TableCell>
                                    <TableCell>{result.labels.join(', ')}</TableCell>
                                    <TableCell>{result.results.join(', ')}</TableCell>
                                    <TableCell>{result.binary_results.join(', ')}</TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </Box>
                        </TableCell>
                      </TableRow>
                    )}
                  </React.Fragment>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}


    </Container>
  );
};

export default HomeView;
