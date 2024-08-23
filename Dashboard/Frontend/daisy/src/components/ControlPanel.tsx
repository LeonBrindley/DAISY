import React, { useEffect, useState } from 'react';
import { Container, Typography, Button, Box, Paper, Grid } from '@mui/material';
import { styled } from '@mui/system';
import { useRecoilState } from 'recoil';
import {selectetedMapLabelState} from '../states/atoms/index'
import ImageDatabaseDash from './ImageDatabaseDash';
import { RecoilRoot } from 'recoil';
import GrassIcon from '@mui/icons-material/Grass';
import SpaIcon from '@mui/icons-material/Spa';
import TerrainIcon from '@mui/icons-material/Terrain';
import EcoIcon from '@mui/icons-material/Nature';
import DirtIcon from '@mui/icons-material/Texture';
import AgricultureIcon from '@mui/icons-material/Agriculture';


const StyledButton = styled(Button)(({ theme }) => ({
  margin: '10px 0',
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'scale(1.05)',
  },
  border: NaN
}));


const ControlPanel = () => {
  const [selectedLayer, setSelectedLayer] = React.useState('species');
  
  const [mapLabel, setMapLabel] = useRecoilState(selectetedMapLabelState);

  const handleLayerChange = (layer: string) => {
    setSelectedLayer(layer);
    setMapLabel(layer);
  };

  useEffect(() => {
    setSelectedLayer(mapLabel as string);
  }, [mapLabel]);

  const getButtonStyle = (layer: string): React.CSSProperties => {
    switch (layer) {
      case 'grass':
        return selectedLayer === 'grass' ? { backgroundColor: 'rgb(19,109,21)', color: 'white' } : {};
      case 'clover':
        return selectedLayer === 'clover' ? { backgroundColor: 'rgb(255, 214, 0)', color: 'black' } : {};
      case 'soil':
        return selectedLayer === 'soil' ? { backgroundColor: 'rgb(139, 69, 19)', color: 'white' } : {};
      case 'dung':
        return selectedLayer === 'dung' ? { backgroundColor: 'rgb(255, 69, 58)', color: 'white' } : {};
      default:
        return {};
    }
  };

  return (
    <Paper elevation={4} sx={{ padding: '20px', margin: '20px', backgroundColor: '#f5f5f5', borderRadius: '10px' }}>
      <Typography variant="h4" gutterBottom>
        Species Distribution
      </Typography>
      <Typography variant="h6">Choose a category to analyze</Typography>

      <StyledButton
        variant={selectedLayer === 'species' ? 'contained' : 'outlined'}
        color="primary"
        onClick={() => handleLayerChange('species')}
        startIcon={<EcoIcon />}
      >
        Species Distribution
      </StyledButton>

      <StyledButton
        variant={selectedLayer === 'topography' ? 'contained' : 'outlined'}
        color="primary"
        onClick={() => handleLayerChange('topography')}
        startIcon={<TerrainIcon />}
      >
        Topography
      </StyledButton>

      <StyledButton
        variant={selectedLayer === 'soil' ? 'contained' : 'outlined'}
        color="primary"
        onClick={() => handleLayerChange('soil')}
        startIcon={<DirtIcon />}
      >
        Soil Class
      </StyledButton>

      <Typography variant="h6" sx={{ marginTop: '20px' }}>
        Choose a species to visualize
      </Typography>

      <StyledButton
        variant="outlined"
        onClick={() => handleLayerChange('grass')}
        style={getButtonStyle('grass')}
        startIcon={<GrassIcon />}
      >
        Grass
      </StyledButton>

      <StyledButton
        variant="outlined"
        onClick={() => handleLayerChange('clover')}
        style={getButtonStyle('clover')}
        startIcon={<SpaIcon />}
      >
        Clover
      </StyledButton>

      <StyledButton
        variant="outlined"
        onClick={() => handleLayerChange('soil')}
        style={getButtonStyle('soil')}
        startIcon={<DirtIcon />}
      >
        Bare Soil
      </StyledButton>

      <StyledButton
        variant="outlined"
        onClick={() => handleLayerChange('dung')}
        style={getButtonStyle('dung')}
        startIcon={<AgricultureIcon />}
      >
        Dung
      </StyledButton>
    </Paper>
  );
};

export default ControlPanel;