const canvas1 = document.getElementById('barGraph1');
const ctx1 = canvas1.getContext('2d');

const canvas2 = document.getElementById('barGraph2');
const ctx2 = canvas2.getContext('2d');

// 데이터
const data1 = [50];
const data2 = [80];

// 초기 설정
const barWidth = 45;
const barSpacing = 20;
const maxBarHeight = 200;
const animationDuration = 1000; // in milliseconds
const framesPerSecond = 60;
const frames = animationDuration / 1000 * framesPerSecond;
const barColors1 = ['#007bff'];
const barColors2 = ['#ff5733'];

let currentFrame = 0;

function drawBar(ctx, x, height, color) {
  ctx.fillStyle = color;
  ctx.fillRect(x, canvas1.height - height, barWidth, height);
}

function drawDualGraph() {
  ctx1.clearRect(0, 0, canvas1.width, canvas1.height);
  ctx2.clearRect(0, 0, canvas2.width, canvas2.height);

  const startX = (canvas1.width - (barWidth + barSpacing) * data1.length + barSpacing) / 2;

  for (let i = 0; i < data1.length; i++) {
    const barHeight1 = (currentFrame / frames) * data1[i] * (maxBarHeight / 100);
    const barHeight2 = (currentFrame / frames) * data2[i] * (maxBarHeight / 100);
    const x = startX + (barWidth + barSpacing) * i;

    drawBar(ctx1, x, barHeight1, barColors1[i]);
    drawBar(ctx2, x, barHeight2, barColors2[i]);
  }

  if (currentFrame < frames) {
    currentFrame++;
    requestAnimationFrame(drawDualGraph);
  }
}

drawDualGraph();
