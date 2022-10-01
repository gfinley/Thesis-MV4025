SVGSetupMarker = {};

(function() {
    
    var _size = 0.5;
    
    SVGSetupMarker.setupMarkerGroup = null;
    SVGSetupMarker.setupMarkerIndex = {};
    SVGSetupMarker.param = null;
    SVGSetupMarker.styleIndex = {"setup-type-blue":"blue", "setup-type-red":"red"};
    
    SVGSetupMarker.init = function(param) {
        SVGSetupMarker.setupMarkerGroup = document.createElementNS(SVG.svgNS, 'g');
        SVG.svg.appendChild( SVGSetupMarker.setupMarkerGroup );
        SVGSetupMarker.param = param;
    }
    
    SVGSetupMarker.setAllVisible = function(value) {
        let vis = "hidden";
        if (value)
            vis = "visible";
        SVGSetupMarker.setupMarkerGroup.setAttributeNS(null, 'visibility', vis);
    }
    
    SVGSetupMarker.addMarker = function(hex, type) {
        if (SVGSetupMarker.setupMarkerIndex[hex.id])
            SVGSetupMarker.removeMarker(hex);
        hex.setup = type;
        let param = SVGSetupMarker.param;
        let position = SVGUtil.gridToSVG(hex.x_grid, hex.y_grid, param.x_hex_margin, param.y_hex_margin, param.width);
        let x = position.x - _size/2;
        let y = position.y - _size/2;
        let elem = document.createElementNS(SVG.svgNS, 'rect');
        elem.setAttributeNS(null, 'x', x);
        elem.setAttributeNS(null, 'y', y);
        elem.setAttributeNS(null, 'width', _size);
        elem.setAttributeNS(null, 'height', _size);
        elem.setAttributeNS(null, 'stroke', "transparent");
        elem.setAttributeNS(null, 'fill', SVGSetupMarker.styleIndex[type]);
        elem.setAttributeNS(null, 'stroke-width', 0);
        elem.setAttributeNS(null, 'pointer-events', "none");
        SVGSetupMarker.setupMarkerIndex[ hex.id ] = elem;
        SVGSetupMarker.setupMarkerGroup.appendChild(elem);
    }
    
    SVGSetupMarker.removeMarker = function(hex) {
        hex.setup = null;
        let marker = SVGSetupMarker.setupMarkerIndex[hex.id];
        if (marker) {
            marker.remove();
            delete SVGSetupMarker.setupMarkerIndex[hex.id];
        }
    }
    
    SVGSetupMarker.removeAllMarkers = function() {
        for (let hexId in SVGSetupMarker.setupMarkerIndex) {
            let hex = Map.hexIndex[hexId];
            SVGSetupMarker.removeMarker(hex);
        }
    }
    
}())