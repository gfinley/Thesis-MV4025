var SVG = {};

(function() { 
    SVG.svgNS = 'http://www.w3.org/2000/svg';

    SVG.pathIndex = {};
    
    SVG.createView = function(param) { 
        SVG.param = param;
        SVG.svg = SVGUtil.recreateMysvg();
        document.body.appendChild(SVG.svg);       
        
        SVG.svg.addEventListener("mouseup",svgMouseUpHandler);
        SVG.svg.addEventListener("mousedown",svgMouseDownHandler);
        
        MapEditorPalette.add(param, SVG.svg);
        MapView.add(param, SVG.svg);
    };
      
}())