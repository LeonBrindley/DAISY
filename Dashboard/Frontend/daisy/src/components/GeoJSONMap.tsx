import React, { useRef, useEffect,useState } from 'react';
import { useRecoilState } from 'recoil';
import { DefaultService } from '../clients/general/index';
import Map from '@arcgis/core/Map';
import Graphic from '@arcgis/core/Graphic';
import MapView from '@arcgis/core/views/MapView';
import GeoJSONLayer from '@arcgis/core/layers/GeoJSONLayer';
import '@arcgis/core/assets/esri/themes/light/main.css';
import SimpleMarkerSymbol from '@arcgis/core/symbols/SimpleMarkerSymbol';
import Point from '@arcgis/core/geometry/Point';
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer';
import { useRecoilValue, useSetRecoilState } from 'recoil';
import { imagesState, selectedImageIdState , selectedFieldState} from '../states/atoms';
import { inferenceResultsState } from '../states/atoms/inferenceResultsAtom';
import {selectetedMapLabelState} from '../states/atoms/index'
import { Button, Container, Slider,IconButton, Typography,Tooltip, Box, Grid, Card, CardContent, CardHeader , Link} from '@mui/material';
import { InferenceResultSchemaOut } from '../clients/inference';
import InfoIcon from '@mui/icons-material/Info';
import ControlPanel from './ControlPanel';
import GrassIcon from '@mui/icons-material/Grass';
import SpaIcon from '@mui/icons-material/Spa';
import TerrainIcon from '@mui/icons-material/Terrain';
import EcoIcon from '@mui/icons-material/Nature';
import DirtIcon from '@mui/icons-material/Texture';
import AgricultureIcon from '@mui/icons-material/Agriculture';

interface ThresholdSliderProps {
  threshold: number;
  setThreshold: (value: number) => void;
  min: number;
  max: number;
  step: number;
  name: 'Lat' | 'Long' | 'Threshold'; // Specify the possible values for `name`
}

const ThresholdSlider: React.FC<ThresholdSliderProps> = ({ threshold, setThreshold, min, max, step, name}) => {
  const handleSliderChange = (event: Event, newValue: number | number[]) => {
    setThreshold(newValue as number);
  }; 


  const getLabelText = (name: 'Lat' | 'Long' | 'Threshold') => {
    switch (name) {
      case 'Lat':
        return `Latitude represents the geographical coordinate that specifies the north-south position of a point on the Earth's surface. It is typically measured in degrees, with values ranging from -90° at the South Pole to +90° at the North Pole.`;
      case 'Long':
        return `Longitude is the geographical coordinate that specifies the east-west position of a point on the Earth's surface. It is measured in degrees, with values ranging from -180° to +180°.`;
      case 'Threshold':
      default:
        return 'Threshold represents a probability used to determine the presence of a certain class (grass, clover, soil, dung). For instance, if the threshold is set to 0.75, it means that the model needs to have at least a 75% probability that a particular class exists.';
    }
  };

  return (
    <Box sx={{ width: 300, marginBottom: 0, display: 'flex', alignItems: 'center' }}>
      <Tooltip
        title={
          <Box>
            <Typography variant="body2" sx={{ fontSize: '0.7rem' }}>
              {getLabelText(name)}
            </Typography>
          </Box>
        }
        arrow
        placement="right"
      >
        <IconButton sx={{ padding: '4px' }}>
          <InfoIcon sx={{ fontSize: '1rem' }} color="primary" />
        </IconButton>
      </Tooltip>

      <Typography variant="h6" sx={{ fontSize: '0.65rem', marginLeft: 1 }} gutterBottom>
        {name} ({threshold.toFixed(2)})
      </Typography>

      <Slider
        value={threshold}
        onChange={handleSliderChange}
        min={min}
        max={max}
        step={step}
        aria-labelledby="threshold-slider"
        sx={{ color: 'primary.main', marginLeft: 2, '& .MuiSlider-thumb': { width: 12, height: 12 }, '& .MuiSlider-track': { height: 4 }, '& .MuiSlider-rail': { height: 4 } }}
      />
    </Box>

  );
};


const fieldNameLookup: { [key: number]: string } = {
  20: "Diary_Corner",
  15: "Dairy_East",
  6: "Dairy_South",
  7: "Lower_Wheaty",
  13: "Dairy_North"
  // Add more mappngs as needed
};

const GeoJSONMap: React.FC = () => {
  const mapDiv = useRef<HTMLDivElement>(null);
  const images = useRecoilValue(imagesState); // Get the images from Recoil state
  const setSelectedImageId = useSetRecoilState(selectedImageIdState); // Set the selected image ID
  const setSelectedField = useSetRecoilState(selectedFieldState);
  const selectedField = useRecoilValue(selectedFieldState);
  const [mapLabel, setMapLabel]= useRecoilState(selectetedMapLabelState);
  const [inferenceResults, setInferenceResults] = useRecoilState(inferenceResultsState);
  const [threshold, setThreshold] = useState<number>(0.9); // Initial threshold value
  const [xCoordinate, setxCoordinate] = useState<number>(-3.89711021814296);
  const [yCoordinate, setyCoordinate] = useState<number>(50.7669670160681);

  useEffect(() => {
    if (mapDiv.current) {
      const map = new Map({
        basemap: 'topo-vector',
      });

      const view = new MapView({
        container: mapDiv.current,
        map: map,
        center: [xCoordinate, yCoordinate],
        zoom: 17,
      });

      const graphicsLayer = new GraphicsLayer();
      map.add(graphicsLayer);


      inferenceResults.forEach((inferenceResult) => {

        const image = images.find((image) => image.img_url === inferenceResult.img_id)

        if (image) {
          var point_color = [0, 0, 0];
          var i = 0;

          if (mapLabel == "soil") {
            point_color = [139, 69, 19];
            i = 2;
          } else if (mapLabel == "grass") {
            point_color = [19,109,21];
            i= 0;
          } else if (mapLabel == "dung") {
            point_color = [255, 69, 58];
            i = 3;
          } else {
            point_color = [255, 214, 0];
            i = 1;
          }


        if (inferenceResult.results[i] > threshold) {
          const point = new Point({
            longitude: image.coordinates[1],
            latitude: image.coordinates[0],
  
          });

          const markerSymbol = new SimpleMarkerSymbol({
            color: point_color, // RGB color
            outline: {
              color: point_color, // white outline
              width: 100,
            },
          })

          const pointGraphic = new Graphic({
            geometry: point,
            symbol: markerSymbol,
            attributes: { ...image },
            popupTemplate: {
              title: "Image Details",
              content: `<p>ID: ${image.id}</p><p>Coordinates: ${image.coordinates.join(', ')}</p><p>Time: ${image.time}</p>`,
              actions: [
                {
                  id: "select-image",
                  title: "Select Image",
                  type: "button"
                }
              ]
            },
          });
  
          graphicsLayer.add(pointGraphic);
        }
      }
      })

      images.forEach((image) => {
        const point = new Point({
          longitude: image.coordinates[1], // assuming long, lat order
          latitude: image.coordinates[0],
        });
        const markerSymbol = new SimpleMarkerSymbol({
          color: [226, 119, 40], // RGB color
          outline: {
            color: [255, 255, 255], // white outline
            width: 0,
          },
          size: 5
        });
        const pointGraphic = new Graphic({
          geometry: point,
          symbol: markerSymbol,
          attributes: { ...image },
          popupTemplate: {
            title: "Image Details",
            content: `<p>ID: ${image.id}</p><p>Coordinates: ${image.coordinates.join(', ')}</p><p>Time: ${image.time}</p>`,
            actions: [
              {
                id: "select-image",
                title: "Select Image",
                type: "button"
              }
            ]
          },
        });

        view.on("click", (event) => {
          view.hitTest(event).then((response) => {
            const results = response.results;
            if (results.length > 0) {
              const graphicHit = results.find(result => (result as __esri.GraphicHit).graphic?.layer === graphicsLayer) as __esri.GraphicHit;
              if (graphicHit && graphicHit.graphic) {
                const attributes = graphicHit.graphic.attributes;
                if (attributes && attributes.id) {
                  setSelectedImageId(attributes.id);
                }
              }
            }
          });
        });

        graphicsLayer.add(pointGraphic);
      });


       view.when(() => {

        const geojsonUrl = 'https://b6627f742a11.ngrok.app/GJSON.geojson';
        const blueSymbol = {
            type: "simple-fill",
            color: "rgba(0, 0, 255, 0.1)",
            outline: {
              color: "blue",
              width: 1
            }
          } as __esri.SimpleFillSymbolProperties;

          const greenSymbol = {
            type: "simple-fill",
            color: "rgba(0, 255, 0, 0.1)",
            outline: {
              color: "green",
              width: 1
            }
          } as __esri.SimpleFillSymbolProperties;
          
          const redSymbol = {
            type: "simple-fill",
            color: "rgba(255, 0, 0, 0.1)",
            outline: {
              color: "red",
              width: 1
            }
          } as __esri.SimpleFillSymbolProperties;


        const blueIDs = [1, 2, 3, 11, 6, 15, 20];
        const greenIDs = [14, 16, 9, 19, 8, 10, 7];
        const redIDs = [23, 22, 21, 24, 26, 4, 13,5, 27, 28, 12];

        const uniqueValueInfos = [
          ...blueIDs.map(id => ({ value: id, symbol: blueSymbol })),
          ...greenIDs.map(id => ({ value: id, symbol: greenSymbol })),
          ...redIDs.map(id => ({ value: id, symbol: redSymbol })),
        ];

        const geojsonLayer = new GeoJSONLayer({
          url: geojsonUrl,
          renderer: {
            type: "unique-value",
            field: "ET_ID", // Replace with a unique field from your data
            defaultSymbol: {
              type: "simple-fill",
              color: "rgba(128, 128, 128, 0.1)",
              outline: {
                color: "blue",
                width: 1
              }
            },
            uniqueValueInfos: uniqueValueInfos
          } as __esri.UniqueValueRendererProperties,
          popupTemplate: {
            title: "{Field_Name}", // Replace with a relevant field from your data
            content: '<p>Field ID: {ET_ID}</p><p>Area: {Area_ha} ha</p>'
          
          }
       });


        map.add(geojsonLayer);


        view.on("click", (event: any) => {

          view.hitTest(event).then((response: any) => {
              const results = response.results;
              if (results.length > 0) {
                const graphicHit = results.find((result: __esri.GraphicHit) => (result as __esri.GraphicHit).graphic?.layer === geojsonLayer) as __esri.GraphicHit;
                if (graphicHit && graphicHit.graphic) {
                  const attributes = graphicHit.graphic.attributes;

                  if (attributes && attributes.ET_ID) {
                    console.log(attributes.ET_ID)
                    console.log("Field Name:", fieldNameLookup[attributes.ET_ID] || `Field ID: ${attributes.ET_ID}`);
                    setSelectedField(fieldNameLookup[attributes.ET_ID] as string)
                  }

                }
              }
          });
        });
          
        });
      



      return () => {
        if (view) {
          view.destroy();
        }
      };
    }
  }, [mapLabel, threshold, images, setSelectedImageId, xCoordinate, yCoordinate, setSelectedField]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Title Section */}
      <Typography variant="h4" sx={{ marginLeft: 5 }} gutterBottom>
        Daisy: Map
      </Typography>

      {/* Main Content Section */}
      <Box sx={{ display: 'flex', flex: 1 }}>
        {/* Map Section */}
        <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }} ref={mapDiv}>
          {/* You can place any overlay elements here if needed */}
          <Box
            sx={{
              position: 'absolute',
              top: 20,
              left: 20,
              width: { xs: '90%', sm: 400 },
              backgroundColor: '#fff',
              boxShadow: 3,
              zIndex: 1000,
              borderRadius: 2,
            }}
          >
            <Card>
              <CardHeader
                title={
                  <Typography variant="h6" sx={{ fontSize: '0.875rem' }}> {/* Adjust the font size here */}
                    Controls
                  </Typography>
                }
              />
              <CardContent sx={{ fontSize: '0.75rem' }}> {/* Adjust the font size in the content */}
                <ThresholdSlider
                  name="Threshold"
                  threshold={threshold}
                  setThreshold={setThreshold}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <ThresholdSlider
                  name="Lat"
                  threshold={xCoordinate}
                  setThreshold={setxCoordinate}
                  min={-3.90}
                  max={-3.89}
                  step={0.0001}
                />
                <ThresholdSlider
                  name="Long"
                  threshold={yCoordinate}
                  setThreshold={setyCoordinate}
                  min={50.76}
                  max={50.77}
                  step={0.0001}
                />
              </CardContent>
            </Card>
          </Box>
        </Box>

        {/* Side Panel Section */}
        <Box
          sx={{
            width: { xs: '100%', sm: 400 }, // Fixed width for the side panel
            backgroundColor: '#fff',
            boxShadow: 3,
            display: 'flex',
            flexDirection: 'column',
            overflowY: 'auto', // Scrollable if content overflows
          }}
        >
          {/* Control Panel Card */}
          <Card sx={{ marginBottom: 3, flexShrink: 0 }}>
            <CardContent>
              <ControlPanel />
            </CardContent>
          </Card>
          {selectedField &&
          <Card sx={{ flexGrow: 1, overflow: 'auto', flexShrink: 0 }}>
            <CardContent>
              <SelectedFieldDetails/>
            </CardContent>
          </Card>  
          }      

          {/* Selected Image Details Card */}
          <Card sx={{ flexGrow: 1, overflow: 'auto', flexShrink: 0 }}>
            <CardContent>
            
              <SelectedImageDetails />
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Box>
  );
};

export default GeoJSONMap;


const SelectedFieldDetails: React.FC = () => {
  const selectedField = useRecoilValue(selectedFieldState);
  const [numImagesField, setNumImagesField] = useState<number>(0);
  const [nitrogenQuantity, setNitrogenQuantity] = useState<string>('');
  const [inferenceResults, setInferenceResults] = useRecoilState(inferenceResultsState);
  const [averageCoverage, setAverageCoverage] = useState<number[]>([]);

  const reloadFieldDetails = (field: string | null) => {


    return
  }

  useEffect(() => {
    reloadFieldDetails(selectedField);

    const groupedResults = inferenceResults.reduce((acc: any, result: any) => {
      if (!acc[result.img_field]) {
        acc[result.img_field] = [];
      }
      acc[result.img_field].push(result);

      return acc
    }, {}); 

    const fieldResults = selectedField ? groupedResults[selectedField] : [];

    if (fieldResults) {

      const numElements = fieldResults.length;
      const numCategories = 4;

      const coverageSums = new Array(numCategories).fill(0);

          // Sum up the coverage for each category
      fieldResults.forEach((result: { percentage_coverage: any[]; }) => {
        result.percentage_coverage.forEach((coverage: any, index: any) => {
          coverageSums[index] += coverage;
        });
      });

      // Calculate the average for each category
      const averages = coverageSums.map(sum => sum / numElements);

      if (averages[1] < 20) {
        setNitrogenQuantity('Not Enough Clover')
      } else if (averages[1] >20 && averages[1] <30) {
        setNitrogenQuantity('180 kg N/ha')
      } else if (averages[1] >30) {
        setNitrogenQuantity('>180 kg N/ha')
      } else if (averages[1] >40) {
        setNitrogenQuantity('>240 kg N/ha')
      } else if (averages[1] >50) {
        setNitrogenQuantity('>300 kg N/ha')
      }


      setAverageCoverage(averages);
      setNumImagesField(fieldResults.length)
      console.log(averages);
      console.log(fieldResults);
    } else {
      setNitrogenQuantity('No Images to analyse')
      setAverageCoverage([]);
      setNumImagesField(0);
    }
  }, [selectedField]);

  return (
    <Card>
      <CardHeader
        title={
          <Box display="flex" flexDirection="column" alignItems="start">
            <Typography variant="h6">
              <strong>Selected Field: {selectedField}</strong>
            </Typography>
            <Typography variant="body1">
              Number of images: {numImagesField}
            </Typography>
            {(averageCoverage) &&
            <Box>
            <Box display="flex" alignItems="center">
              <GrassIcon style={{ color: 'rgb(19, 109, 21)' }} /> 
              <Typography>
                {typeof averageCoverage[0] === 'number' ? averageCoverage[0].toFixed(2) : 'N/A'}% Grass
              </Typography>
            </Box>
            <Box display="flex" alignItems="center">
              <SpaIcon style={{color: 'rgb(255, 214, 0)'}}/> 
              <Typography>
              {typeof averageCoverage[1] === 'number' ? averageCoverage[1].toFixed(2) : 'N/A'}% Clover
              </Typography>
            </Box>
            <Box display="flex" alignItems="center">
              <DirtIcon style={{color: 'rgb(139, 69, 19)'}}/> 
              <Typography>
              {typeof averageCoverage[2] === 'number' ? averageCoverage[2].toFixed(2) : 'N/A'}% Soil
              </Typography>
            </Box>
            <Box display="flex" alignItems="center">
              <AgricultureIcon style={{color: 'rgb(255, 69, 58)'}}/> 
              <Typography>
              {typeof averageCoverage[3] === 'number' ? averageCoverage[3].toFixed(2) : 'N/A'}% Dung
              </Typography>
            </Box>
            </Box>
            }
            <Box display="flex" alignItems="center">
            <Tooltip
              title={
                <Box>
                  <Typography variant='body2'>
                    <strong>Top Tips:</strong>
                  </Typography>
                  <Typography variant="body2">
                    - Aim for 30% clover content in sward for optimal yields.
                  </Typography>
                  <Typography variant="body2">
                    - White clover can increase yields by up to 15%, thanks to its nitrogen-fixing ability and perennial nature.
                  </Typography>
                  <Typography>
                    For more info, <Link href="https://projectblue.blob.core.windows.net/media/Default/Beef%20&%20Lamb/ManagingClover2871_210310_WEB.pdf" target="_blank" rel="noopener noreferrer" style={{ color: 'white' }}>click here</Link>.
                  </Typography>
                </Box>
              }
              arrow
            >
              <IconButton>
                <InfoIcon color="primary" fontSize='small'/>
              </IconButton>
            </Tooltip>
              <Typography variant="subtitle1" component="span">
                <strong>Nitrogen Insights</strong>
              </Typography>
            </Box>
            <Typography variant="body1">
              Predicted Nitrogen: {nitrogenQuantity}
            </Typography>

          </Box>
        }
      />
    </Card>
  );
};


// Change this object to be 
// Selected Field: {selectedField}
// Number of images: {numImagesField}
// <Icon1></Icon1> {percentageGrass}
// <Icon1></Icon1> {percentageClover}
// <Icon1></Icon1> {percentageSoil}
// <Icon1></Icon1> {percentageDung}
// Nitrogen Insights (i)icon
// Predicted Nitrogen {nitrogenQuantity}Kg/N
// Suggested Action
// {suggestedActionText}



const SelectedImageDetails: React.FC = () => {
  const selectedImageId = useRecoilValue(selectedImageIdState);
  const images = useRecoilValue(imagesState);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState<boolean>(false);
  const [inferenceResults, setInferenceResults] = useRecoilState(inferenceResultsState);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [mapLabel, setMapLabel]= useRecoilState(selectetedMapLabelState);
  const selectedImageDetails = images.find(image => image.id === selectedImageId);



  useEffect(() => {
    const fetchImage = async () => {
      if (selectedImageId) {

        if (showOverlay && selectedImageDetails) {
          const selectedInferenceResult = inferenceResults.find(inference => inference.img_id === selectedImageDetails.img_url);
          if (selectedInferenceResult) {

            const response = await DefaultService.getPredictionImageFileGetPredictionImageFilePredictionIdGet(selectedInferenceResult.id, mapLabel.toString())
            if (response && response.image) {
              setSelectedImage(`data:image/jpeg;base64,${response.image}`);
            }

          } 

        } else {
          const response = await DefaultService.getImageFileGetImageFileImageIdGet(selectedImageId);
          if (response && response.image) {
            setSelectedImage(`data:image/jpeg;base64,${response.image}`);
          }
        }
      }
    };
    fetchImage();
  }, [selectedImageId, showOverlay, mapLabel]);


  if (!selectedImageDetails) {
    return (
      <Typography variant="body1" color="textSecondary">
        No image selected.
      </Typography>
    );
  }


  const groupedResults = inferenceResults.reduce((acc:any, result: any) => {
    if (!acc[result.img_id]) {
      acc[result.img_id] = [];
    }
    acc[result.img_id].push(result);
    return acc;
  }, {});

  return (
    <Card>
         <CardHeader
        title={
          <Box display="flex" alignItems="center">
            {selectedImage ?
            <Tooltip
              title={
                <Box>
                  <Typography variant="body2">
                    <strong>ID:</strong> {selectedImageDetails.id}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Img Name:</strong> {selectedImageDetails.img_url}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Coordinates:</strong> {selectedImageDetails.coordinates.join(', ')}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Time:</strong> {selectedImageDetails.time}
                  </Typography>
                </Box>
              }
              arrow
            >
              <IconButton>
                <InfoIcon color="primary" />
              </IconButton>
            </Tooltip> : <></>
              }
            {selectedImage ?  <Typography>Image Selected</Typography> : <Typography>Select an Image</Typography> }
          </Box>
        }
      />
      {selectedImage &&
      <CardContent>
        <Box display="flex" justifyContent="center">
           
            <img src={selectedImage} alt="Selected" style={{ maxWidth: '100%', height: 'auto', marginBottom: 16 }} />
          
        </Box>
        <Button
          variant="contained"
          color="primary"
          onClick={() => setShowOverlay(!showOverlay)}
          sx={{ marginTop: 2 }}
        >
          {showOverlay ? 'Deactivate Overlay' : 'Activate Overlay'}
        </Button>
      </CardContent>
      }
    </Card>
  );
  //   <div>
  //     <h3>Selected Image Details</h3>
  //     <p>ID: {selectedImageDetails.id}</p>
  //     <p>Img Name: {selectedImageDetails.img_url}</p>
  //     <p>Coordinates: {selectedImageDetails.coordinates.join(', ')}</p>
  //     <p>Time: {selectedImageDetails.time}</p>

  //     <Button onClick={() => {showOverlay ? setShowOverlay(false) : setShowOverlay(true) }}>
  //       {showOverlay ?  "Deactivate" : "Activate Overlay"} 
  //       </Button>
  //     <p>Inference_results</p>


  //       {selectedImage && (
  //         <>
  //           <img src={selectedImage} alt="Selected" style={{ maxWidth: '40%', display: 'block' }} />
  //         </>
  //       )}        
  //   </div>
  // );
};


      {/* {selectedImage && groupedResults[selectedImageDetails.img_url] && groupedResults[selectedImageDetails.img_url].length > 0 && (
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
                              {groupedResults[selectedImageDetails.img_url].map((result: any, idx: any) => (
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
                  )} */}