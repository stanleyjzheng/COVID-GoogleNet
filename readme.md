# COVID-GoogleNet

Pytorch implementation of [COVID-ResNet](https://github.com/Stanley-Zheng/COVID-ResNet) using GoogleNet (Inception v1). Please see COVID-ResNet readme for details in cloning the datset. The purpose of this model is to provide a slightly lower performance, more efficient model to use on low-powered machines and on the web. To see an online demo, please visit [stanleyzheng.ca](https://stanleyzheng.ca). This demo is running on less than 200mb of memory, and the entire slug size (including all dependencies) is 239mb. 

### Results
![Confusion matrix](https://img.techpowerup.org/200630/index906.png)

COVID-ResNet Test:
<div><table>
  <tr>
    <th colspan="2">Sensivity(%)</th>
  </tr>
  <tr>
    <td>Normal</td>
    <td>COVID-19</td>
  </tr>
  <tr>
    <td>96.00</td>
    <td>97.50</td>
  </tr>
</table></div>

<div><table>
  <tr>
    <th colspan="2">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td>Normal</td>
    <td>COVID-19</td>
  </tr>
  <tr>
    <td>95.12</td>
    <td>97.96</td>
  </tr>
</table></div>
