var SVGGui = {};

(function() {
    
var unitSymbolDimensions = null;
var hexMarks = [];

function setUnitSymbolDimensions() {
    // Use first unit that's still alive and on the map
    let id = null;
    for (const unit of Unit.units) {
        if (!unit.ineffective && unit.hex) {
            id = unit.uniqueId;
            break;
        }
    } 
    let elem = document.getElementById(id);
    let bbox = SVGUtil.getTransformedBBox(elem);
    unitSymbolDimensions = {width:bbox.width, height:bbox.height};
}

SVGGui.markHex = function(hex, color, id) {
    if (!unitSymbolDimensions)  setUnitSymbolDimensions();
    let {width,height} = unitSymbolDimensions;
    let usd = unitSymbolDimensions;
    let hexElem = document.getElementById(hex.id);
    let bbox = SVGUtil.getTransformedBBox(hexElem);
    let [x_center, y_center] = [bbox.x + bbox.width/2, bbox.y + bbox.height/2];
    let mg = 0.05 * width;
    selectionMarker = SVGUtil.makeRect(x_center-width/2-mg,y_center-height/2-mg,width+2*mg,height+2*mg,"transparent",color);
    selectionMarker.setAttributeNS(null, 'id', id);
    selectionMarker.addEventListener("mousedown",markerMouseDown);
    SVG.svg.appendChild(selectionMarker);
    hexMarks.push(selectionMarker);
}

SVGGui.clearMarks = function() {
    for (let mark of hexMarks)
        mark.remove();
}

}());