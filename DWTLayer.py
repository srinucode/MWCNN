
class DWTLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super(DWTLayer, self).__init__()
        self.wavelet = wavelet
        print("Hello world")

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        outputs = []
        output = []
        
        # Apply DWT to each channel independently
        for i in range(channels):
            img = x[:, i, :, :].detach().cpu().numpy()  # Extract the i-th channel and convert to numpy
            img = img.squeeze(0)  # Remove batch dimension for DWT
            
            
            # Perform DWT on the 2D image
            coeffs = pywt.dwt2(img, self.wavelet)
            cA, (cH, cV, cD) = coeffs
            
            # Stack coefficients
            cA = np.expand_dims(cA, axis=0)  # Add channel dimension
            cA = cA.squeeze(0)
            output.append(cA)
            cH = np.expand_dims(cH, axis=0)
            cH = cH.squeeze(0)
            output.append(cH)
            cV = np.expand_dims(cV, axis=0)
            cV = cV.squeeze(0)
            output.append(cV)
            cD = np.expand_dims(cD, axis=0)
            cD = cD.squeeze(0)
            output.append(cD)
       
            
            
        
        # Stack all channels along the batch dimension
        output = np.stack(output, axis=0)
        outputs.append(output)
        #print(outputs.shape)
        output = torch.tensor(outputs, dtype=x.dtype, )
        
        return output
