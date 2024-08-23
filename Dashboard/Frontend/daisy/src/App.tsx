import React, { useState } from 'react';
import { Container, Box, Grid, Drawer, List, ListItem, ListItemText, ListItemIcon, Typography } from '@mui/material';
import { styled } from '@mui/system';
import GeoJSONMap from './components/GeoJSONMap';
import ImageDatabaseDash from './components/ImageDatabaseDash';
import { RecoilRoot } from 'recoil';
import ControlPanel from './components/ControlPanel';
import DashboardIcon from '@mui/icons-material/Dashboard';
import MapIcon from '@mui/icons-material/Map';
import HomeIcon from '@mui/icons-material/Home';
import HomeView from './components/HomeView';



const App = () => {
  const [currentView, setCurrentView] = useState('home'); // 'home', 'dashboard', or 'map'

  return (
    <RecoilRoot>
      <Drawer
        variant="permanent"
        anchor="left"
        PaperProps={{
          style: { width: '240px', backgroundColor: '#f4f4f4' },
        }}
      >
        <Box display="flex" flexDirection="column" alignItems="center" padding={2}>
          {/* Logo */}
          <img src="/cropped_logo.png" alt="Logo" style={{ width: '150px', marginBottom: '20px' }} />

          {/* Navigation Links */}
          <List>
            <ListItem button onClick={() => setCurrentView('home')}>
              <ListItemIcon>
                <HomeIcon />
              </ListItemIcon>
              <ListItemText primary="Home" />
            </ListItem>

            <ListItem button onClick={() => setCurrentView('map')}>
              <ListItemIcon>
                <MapIcon />
              </ListItemIcon>
              <ListItemText primary="Map & Control" />
            </ListItem>

            <ListItem button onClick={() => setCurrentView('dashboard')}>
              <ListItemIcon>
                <DashboardIcon />
              </ListItemIcon>
              <ListItemText primary="Developer Dashboard" />
            </ListItem>
          </List>
        </Box>
      </Drawer>

      <Container maxWidth="lg" style={{ marginLeft: '240px', paddingTop: '20px' }}>
        {currentView === 'home' && <HomeView />}
        {currentView === 'dashboard' && <ImageDatabaseDash />}
        {currentView === 'map' && (
          <GeoJSONMap />

        )}
      </Container>
    </RecoilRoot>
  );
    // <RecoilRoot>
    //   <Container maxWidth="lg" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
    //     <Box display="flex" justifyContent="center" marginBottom={2}>
    //       <Button
    //         variant="contained"
    //         color={showRequestTest ? 'primary' : 'inherit'}
    //         onClick={() => setShowRequestTest(true)}
    //         style={{ margin: '0 10px' }}
    //       >
    //         Database Dashboard
    //       </Button>
    //       <Button
    //         variant="contained"
    //         color={showRequestTest ? 'inherit' : 'primary'}
    //         onClick={() => setShowRequestTest(false)}
    //         style={{ margin: '0 10px' }}
    //       >
    //         Map & Control
    //       </Button>
    //     </Box>

    //     {showRequestTest ? (
    //       <ImageDatabaseDash />
    //     ) : (
    //       <Grid container spacing={3}>
    //         <Grid item xs={16} md={10}>
    //           <Box style={{ height: '100vh'}}>
    //             <GeoJSONMap />
    //           </Box>
    //         </Grid>
    //         <Grid item xs={16} md={6}>
    //           <ControlPanel />
    //         </Grid>
    //       </Grid>
    //     )}
    //   </Container>
    // </RecoilRoot>

};

export default App;

